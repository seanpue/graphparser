import networkx as nx
from collections import namedtuple,defaultdict
import re
import unicodedata
import yaml
import itertools

ParserRule = namedtuple('ParserRule',
                        ['production',     # output string
                         'prev_classes',   # classes of preceding tokens (before prev_tokens)
                         'prev_tokens',    # previous tokens (before tokens)
                         'tokens',         # tokens to match
                         'next_tokens',    # tokens to follow (after tokens)
                         'next_classes',   # classes of tokens (after next_tokens)
                         'prev_length',    # length of previous tokens and classes
                         'next_length',    # length of next tokens and classes
                         'match_tokens',   # tokens to match (including prev and next tokens)
                         'tokens_length'   # number of tokens = len(tokens)
                        ])

ParserOutput = namedtuple('ParserOutput', ['matches','output'])

class GraphParser:

    def classes_int(self,classes):
        return [self.token_class_names.index(c) for c in classes]


    def unescape_unicode_charnames(self,s):
        """
        Takes \N{charname} in production rule and turns to Unicode string.
        """
        def get_unicode_char(matchobj):
            s = matchobj.group(0)
            m = re.match(r'\\N{(.+)}',s)
            char = unicodedata.lookup(m.group(1))
            return char
        return re.sub(r'\\N{.+?}',get_unicode_char,s)

    def compare_rules(x,y):
        """
        Compares parser rules x,y and sees which has more tokens or conditions (prev, next)
        """
        diff = 10*(len(y['tokens'])-len(x['tokens']))
        if diff != 0:
            return diff
        (x_conds, y_conds) = (0,0)
        for cond in ('prev','next'):
            if cond in x: x_conds +=len(x[cond])
            if cond in y: y_conds +=len(y[cond])
        for cond in ('prev_classes', 'next_classes'):
            if cond in x: x_conds += len(x[cond])
            if cond in y: y_conds += len(y[cond])
        return y_conds - x_conds # see if one has  more <classes>

    def get_onmatch_rules_from_yaml_data(self,rules_raw):
        ''' Some quick code to generate the onmatch rules. It only relies on classes.
            returns a tuple ( (left_classes,right_classes) , prod)
            rules are classes
        '''
        onmatch_rules = [] # clean list of rule

        match_rules = []
        for key in rules_raw:
            if len(key)!=1:
                print 'Could not find key',key
                assert len(key)==1
            rule = key.keys()[0]
            prod_orig = key[rule]
            prod = self.unescape_unicode_charnames(prod_orig)
            m= re.match('([^+]+)\+([^+]+)$',rule)
            assert m
            l = m.group(1) #left
            r =m.group(2)  #right
            cl_l=re.findall('(?<=<)[^<]+(?=>)',l)
            cl_r=re.findall('(?<=<)[^<]+(?=>)',r)
            onmatch_rules.append(( (cl_l, cl_r) , prod ))
        return(onmatch_rules)

    def get_onmatch_rules_by_token(self, omr):
        '''
        Sorts onmatch rules by current match token of rule, reducing number of iterations
        '''
        t_om = defaultdict(list)
        tokens = self.tokens
        for t,t_classes in tokens.iteritems():
            for o in omr:
                match_rules,prod = o
                curr_class=match_rules[1]
                if curr_class[0] in t_classes:
                    t_om[t].append(o)
        return t_om

    def get_onmatch_rules_token_matrix(self, omr):
        '''
        Returns a dictionary of current match token, holding a dictionary of previous tokens that
        contain onmatch rules

        Could switch to indexes. this is an attempted speed up by reducing number of iterations.
        Need to check that if that is in fact the case.
        '''
        token_matrix = {t:[] for t in self.tokens}
        ttm = {t:{tc:[] for tc in self.tokens} for t in self.tokens}
        for t,t_classes in self.tokens.iteritems():
            for o in omr:
                match_rules,prod = o
                curr_class=match_rules[1]
                if curr_class[0] in t_classes:
                    for t2,t2_classes in self.tokens.iteritems():
                        prev_class = match_rules[0]
                        if prev_class[0] in t2_classes:
                            ttm[t][t2].append(o) # was omr - error??
        return ttm

    def rules_from_yaml_data(self,rules_raw):
        """
        Returns sorted and usable parser rules from yaml file.

        rules_raw is a dictionary (loaded from yaml):
            key: <previous token class> token token <next token class>
            value: production

        previous class and next class of token are optional.
        The production allows \N{unicode char name} strings.

        Output of return is list of rules containing:
            prev_class: previous token class [optional, will look behind prev]
            prev:       previous tokens [optional]
            tokens:     list of tokens
            next:       next token class [optional]
            next_class: next token class [option, will look ahead of next]
            production: string production

        """
        rules = [] # clean list of rul

        for key in rules_raw:

            rule = {}           #1       #2        #3

            match_rule = '(?:\(((?:\s?<.+?>\s+)+)?(.+?)\s?\) |((?:\s?<.+?>\s+)+)?)?'#1#2#3

            match_rule += '(.+?)'
            match_rule += '(?:'
            match_rule += '\s+(?:\((.+?)((?:\s+<.+?>?)?)\)$)'
            match_rule += '|'
            match_rule += '((?:\s+<.+?>)+)'
            match_rule += '|'
            match_rule += '$)'

            m = re.match (match_rule, key, re.S)
            assert (m is not None)
            if m.group(1): rule['prev_classes'] = re.findall('<(.+?)>',m.group(1))
            if m.group(2): rule['prev_tokens'] = m.group(2).split(' ')
            if m.group(3): rule['prev_classes'] =  re.findall('<(.+?)>',m.group(3))
            if m.group(4)==' ':
                rule['tokens'] = [' ']
            else:
                rule['tokens'] = [x for x in m.group(4).split(' ') if x!='']
            if m.group(5):
                rule['next_tokens'] = m.group(5).split(' ')
            if m.group(6): rule['next_classes'] = re.findall('<(.+?)>',m.group(6))
            if m.group(7): rule['next_classes'] = re.findall('<(.+?)>',m.group(7))

            rule['production'] = self.unescape_unicode_charnames(rules_raw[key])
            rules.append(rule)

        return rules

    def get_token_match_re_string(self):
        '''
        Generate regular expression string from Parser tokens sorted by length
        Adds final "." in case nothing found
        '''
        tokens = self.tokens.keys()
        sorted_tokens = sorted(tokens, key=len, reverse=True)
        escaped_tokens = map(re.escape, sorted_tokens)
        tokens_re_string = '|'.join(escaped_tokens)+'|.'
        return tokens_re_string

    def generate_token_match_re(self):
        '''
        Create regular expression from Parser.tokens sorted by length
        Adds final "." in case nothing found
        '''
        return re.compile(self.get_token_match_re_string(), re.S)


    def __init__(self, yaml_file='', data=None, blank=' '):
        assert data == None # not implemented yet

        data = yaml.load(file(yaml_file))

        self.dict_rules = self.rules_from_yaml_data(data['rules'])
        self.rules = self.rules_to_tuple(self.dict_rules)

        self.tokens = data['tokens']

        assert len(set(self.tokens))==len(self.tokens) # make sure there are no repeated tokens

        self.token_class_names = sorted(set(itertools.chain(*self.tokens.values())))
        self.token_keys = sorted(self.tokens)

        # check to make sure all tokens in rules exist
        for r in self.rules:
            for tkns in [x for x in r.prev_tokens, r.tokens, r.next_tokens if x!=None]:
                for t in tkns:
                    if not t in self.tokens:
                        print 'Error! Token',t,'not found in rule',r,'in ',yaml_file
                    assert t in self.tokens

        self.token_match_re = self.generate_token_match_re()


        onmatch = data.get('onmatch')
        self.onmatch_rules = None
        if onmatch:
            self.onmatch_rules = self.get_onmatch_rules_from_yaml_data(onmatch)
            self.onmatch_rules_by_token = self.get_onmatch_rules_by_token(self.onmatch_rules)#None
            self.onmatch_rules_token_matrix = self.get_onmatch_rules_token_matrix(self.onmatch_rules)
        DG= self.make_graph()
        self.sorted_out_edges = self.get_sorted_out_edges(DG) # edges are arranged by node and then by weight
        self.sorted_out_edges_by_next_tokens,\
        self.sorted_out_edges_no_tokens=self.get_sorted_out_edges_by_next_tokens(DG)
        self.DG = DG
        self.blank = blank # what token is assumed at before and after tokens, presumably ' '

    def rules_to_tuple(self,rules):
        '''
        Converts rule into a ParserRule named tuple, adding match_tokens, prev_ and next_ length
        '''
        new_rules = []
        for r in rules:
            left = 0
            for x in ['prev_tokens']:#, 'prev_classes']:
                if x in r:
                    left+=len(r[x])
            right=0
            for x in ['tokens', 'next_tokens']:#,'next_classes']:
                if x in r:
                    right+=len(r[x])
            match_tokens=[]
            for x in ['prev_tokens','tokens','next_tokens']:
                if r.get(x):
                    match_tokens.extend(r.get(x))
            pr = ParserRule(production = r['production'],
                            prev_classes = r.get('prev_classes'),#change here
                            prev_tokens = r.get('prev_tokens'),
                            tokens = r['tokens'],
                            next_tokens = r.get('next_tokens'),
                            next_classes = r.get('next_classes'),
                            prev_length = left,
                            next_length = right,
                            match_tokens = match_tokens,
                            tokens_length = len(r['tokens'])#len(tokens)
                            ) # change here
            new_rules.append(pr)
        return tuple(new_rules)

    def tokenize(self,input):
        return self.token_match_re.findall(input)

    def make_graph(self): # turns rules into graph
        DG = nx.DiGraph()
        DG.add_node(0,token=None) #base
        def found_token_at_node(t, n_id):
            for s in DG.successors(n_id):
                if DG.node[s].get('token')==t:
                    return s
            return None

        def weight_of_rule(r):
            '''
            returns a weight of ParserRule r based on number of features
            '''
            weight = 0
            for x in ['prev_classes','prev_tokens','next_classes','next_tokens']:
                if r._asdict().get(x): weight+=1 # treat namedtuple as dict
            return weight

        token_nodes = []
        match_nodes = []
        for rule in self.rules:
            curr_node = 0
            for t in rule.tokens:
                found = found_token_at_node(t,curr_node)
                if found:
                    curr_node = found
                    continue
                else:
                    new_node = len(DG.nodes())
                    DG.add_node(new_node, token=t)
                    token_nodes.append(new_node) # save in list to be used in drawing for labels, color, etc.
                    DG.add_edge(curr_node,new_node,weight=0) # if not a match node, then no weight, so will try first
                    curr_node = new_node

            rule_weight = weight_of_rule(rule) # this will be 1/4 for 4 constraints, 1/3 for 3 constraints, etc.

            new_match_node = len(DG.nodes())
            DG.add_node(new_match_node, rule=rule, found=rule.production)
            match_nodes.append(new_match_node)
            DG.add_edge(curr_node, new_match_node, weight=rule_weight,rule=rule) # for now, just copy the rule
        return DG

    def get_token_classes(self, token):
        '''
        Returns classes of a token
        '''
        return self.tokens[token]

    def get_sorted_out_edges(self,g):
        '''
        Returns dictionary from DG containing list of out edges  sorted by weight
        '''

        out_edges = defaultdict(list)

        for e in g.edges(data=True):
            start,end,data = e
            out_edges[start].append(e)
        sorted_out_edges = defaultdict(list)
        for k,edges in out_edges.iteritems():
            sorted_edges = sorted(edges, key =lambda x: x[2]['weight'],reverse=True)
            sorted_out_edges[k] = sorted_edges
        return sorted_out_edges

    def get_sorted_out_edges_by_next_tokens(self,DG):
        sorted_out_edges_by_next_tokens=defaultdict(lambda:defaultdict(list)) # list of out edges based on nen
        sorted_out_edges_no_tokens=defaultdict(list) # holds rules

        for start_node,edges in self.sorted_out_edges.iteritems():
            for edge in edges:
                start_n,end_nid,edge_data = edge
                end_n = DG.node[end_nid]
                tkn=end_n.get('token')
                if tkn:
                    sorted_out_edges_by_next_tokens[start_n][tkn].append(edge)
                else:
                    sorted_out_edges_no_tokens[start_n].append(edge)
        return sorted_out_edges_by_next_tokens,sorted_out_edges_no_tokens

    def match_rule(self,rule, tkns, t_i, level):
        parser_tokens = self.tokens

        i_start = t_i - rule.prev_length
        if i_start<0 or i_start+len(rule.match_tokens) > len(tkns):
            return False
        if rule.match_tokens != tkns[i_start:i_start+len(rule.match_tokens)]:
            return False
        if rule.prev_classes:
            prev_classes = rule.prev_classes[::-1] # reverse these
            x= i_start - len(prev_classes)
            if x < -1:
                return False
            elif x == -1:
                to_match = [' ']+tkns[i_start+1-len(prev_classes):i_start]
            else:
                to_match = tkns[i_start-len(prev_classes):i_start]
            to_match = to_match[::-1]
            if not all(prev_classes[i] in self.tokens[to_match[i]] for i in range(len(prev_classes))):
                return False
        if rule.next_classes:
            next_classes = rule.next_classes
            end_i = i_start+len(rule.match_tokens)
            if end_i+len(next_classes) > len(tkns)+1:
                return False
            to_match = tkns[end_i:end_i+len(next_classes)] + [self.blank]#[' ']
            if not all(next_classes[i] in self.tokens[to_match[i]] for i in range(len(next_classes))):
                return False
        return True


    def match_first_at(self,tokens, token_i):

        def descend_node(curr_node, level):
            new_edges = []
            if token_i + level < len(tokens):
                curr_token = tokens[token_i+level]
                new_edges = self.sorted_out_edges_by_next_tokens[curr_node][curr_token]
            new_edges = new_edges + self.sorted_out_edges_no_tokens[curr_node]

            for edge in new_edges:#in self.sorted_out_edges[curr_node]:
                next_node = edge[1]
                if edge[2].get('rule'): # if the edge has a rule
                    if self.match_rule(edge[2]['rule'], tokens, token_i,level)==False:
                        continue # skip it
                node_rule = self.DG.node[next_node].get('rule') # matched nodes have rule signalling end?
                if node_rule:
                    return node_rule#self.DG.node[next_node].get('rule')
                if token_i+level < len(tokens): # do not descend if at end of road
                     d =descend_node(next_node, level+1)
                     if d:
                         return d

        return descend_node(0,0)

    def parse(self,string):
        t_i = 0
        tkns = self.tokenize(string)
        if self.onmatch_rules:
            mtkns = [self.blank]+tkns+[self.blank]
        output = ''
        matches = []
        while t_i < len(tkns):
            m = self.match_first_at(tkns, t_i)
            if m==None:
                print "error in string",string,len(string)
            assert m != None # for now, croak on error
            matches.append(m)

            if self.onmatch_rules:
                mt_i = t_i+1
                omr = self.onmatch_rules_token_matrix[ mtkns[mt_i] ][ mtkns[mt_i-1] ]
                if len(omr)>0:
                    for mr in omr:

                        ((l_classes,r_classes),p)=mr

                        if mt_i < len(l_classes) or mt_i+len(r_classes)>len(mtkns):
                            continue

                        l_length = len(l_classes)
                        l_start = mt_i - l_length

                        for i in range(0,len(l_classes)):
                            if not l_classes[i] in self.get_token_classes(mtkns[l_start+i]):
                                continue

                        r_length = len(r_classes)
                        r_start = mt_i + 1

                        for i in range(0,len(r_classes)):
                            if not r_classes[i] in self.get_token_classes(mtkns[r_start+i]):
                                continue

                        output += p
                        break # break out of for loop

            output+=m.production
            t_i += m.tokens_length #len(m.tokens)

        return ParserOutput(output=output,matches=matches)



    def match_all_at(self,tokens, token_i):
        matches = []
        def descend_node(curr_node, level):
            for edge in sorted(list(self.DG.edges(curr_node, data=True)), key=lambda x:x[2]['weight'] ):
                next_node = edge[1]
                if edge[2].get('rule'): # if the edge has a rule
                    if self.match_rule(edge[2]['rule'], tokens, token_i,level)==False:
                        continue
                if self.DG.node[next_node].get('rule'): # matched nodes have found and rule
                            matches.append(self.DG.node[next_node].get('rule'))
                            continue
                if token_i+level < len(tokens):
                        if self.DG.node[next_node].get('token')==tokens[token_i+level]:
                            descend_node(next_node, level+1)
        descend_node(0,0)
        return matches

if __name__ == '__main__':
   p = GraphParser('settings/devanagari.yaml')
   import pdb
   pdb.set_trace()
   print p.parse(' aadmii').output

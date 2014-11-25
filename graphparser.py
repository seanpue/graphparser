# the prev and next classes turned into lists. Had to correct for that
from myparser import Parser # for now, just load yaml rules via Parser
import networkx as nx
from collections import namedtuple

ParserRule = namedtuple('ParserRule', ['production','prev_classes','prev_tokens','tokens','next_tokens','next_classes'])

ParserOutput = namedtuple('ParserOutput', ['matches','output']) 


class GraphParser:

     
    def __init__(self, yaml_file='', data=None, blank=' '):
        
        self.old_parser=Parser(yaml_file)
        rules = self.old_parser.rules
        self.rules = self.rules_to_tuple(rules)
        self.tokens = self.old_parser.tokens
        DG= self.make_graph()
        #print DG
        self.DG = DG
        self.blank = blank # what token is assumed at before and after tokens
        self.onmatch_rules = self.old_parser.onmatch_rules
    
    def rules_to_tuple(self,rules):
        new_rules = []
        for r in rules:
            pr = ParserRule(production = r['production'], 
                            prev_classes = r.get('prev_classes'),#change here
                            prev_tokens = r.get('prev_tokens'),
                            tokens = r['tokens'],
                            next_tokens = r.get('next_tokens'),
                            next_classes = r.get('next_classes')) # change here
            new_rules.append(pr)
        return tuple(new_rules)
        
    def tokenize(self,input):
        return self.old_parser.token_match_re.findall(input)

    def make_graph(self): # turns rules into graph
        DG = nx.DiGraph()
        DG.add_node(0,token=None) #base
        def found_token_at_node(t, n_id):
            for s in DG.successors(n_id):
        #        print s,DG[s]
                if DG.node[s].get('token')==t:
                    return s

            return None

        def weight_of_rule(r): # and here namedtuples get clunky? 
            weight=5 # start at five and subtract.
            if r.prev_classes:
                weight+=-1
            if r.prev_tokens:
                weight+=-1
            if r.next_classes:
                weight+=-1
            if r.next_tokens:
                weight+=-1
            return weight # when we navigate, we will go from low to high in the edges

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
        #            print DG.node[new_node],t
                    DG.add_edge(curr_node,new_node,weight=0) # if not a match node, then no weight, so will try first
                    curr_node = new_node
                
            rule_weight = weight_of_rule(rule) # this will be 1/4 for 4 constraints, 1/3 for 3 constraints, etc.

            new_match_node = len(DG.nodes()) 
            DG.add_node(new_match_node, rule=rule, found=rule.production)
            match_nodes.append(new_match_node)
            DG.add_edge(curr_node, new_match_node, weight=rule_weight,rule=rule) # for now, just copy the rule into the edge and check from that
            #print DG[curr_node][new_match_node]
            #print 'added rule',rule
        return DG

    def match_rule(self,rule, tokens, token_i, level):
#        print "trying to match rule ",rule
        parser_tokens = self.tokens
        # level shows how far we have moved forward from token_i
#        print "trying rule ",rule
#        pdb.set_trace()
        (num_prev_tokens, num_next_tokens) = (0,0)
        if rule.prev_tokens:
            num_prev_tokens = len(rule.prev_tokens)
            start_i = token_i - len(rule.prev_tokens)

            if start_i < 0:
                return False
            for i in range(num_prev_tokens): # had some trouble will all here for some reason
                if not tokens[start_i+i]==rule.prev_tokens[i]:
                    return False
#        print "  passed prev tokens"
        if rule.prev_classes:
#            print 'in prev_class'
            start_i = token_i - num_prev_tokens-1
            prev_token = ''
            if start_i < -1:
                return False
            if start_i == -1: # if just one past, send a 'blank'
                prev_token = self.blank
            else:
                prev_token = tokens[start_i]
            assert len(rule.prev_classes)<2
            if not rule.prev_classes[0] in parser_tokens[prev_token]:
                return False
#        print "passed prev class"
        if rule.next_tokens:
            num_next_tokens = len(rule.next_tokens)
            start_i = token_i + level # 
            if start_i + num_next_tokens > len(tokens):
#                print 'too long'
                return False
            for i in range(num_next_tokens):
                if not tokens[start_i+i]==rule.next_tokens[i]:
                    return False
#        print " pased next tokns"
        if rule.next_classes:
            start_i = token_i+level+num_next_tokens
            if start_i == len(tokens): # if one past
         #       print 'loading blank', self.blank
                next_token = self.blank # load blank
            else:
                next_token = tokens[start_i]
            assert len(rule.next_classes)<2 # only one next clases
            if not rule.next_classes[0] in parser_tokens[next_token]:
#                print 'could not find ', rule.next_classes, 'in ',parser_tokens[next_token], ' were ',parser_tokens
                return False
    #        if start_i + num_next_tokens == len(tokens):  need to deal with this one, but can in next
      #  print "passed next_class"
        return True

    def match_first_at(self,tokens, token_i):
        #print "trying tokens ",tokens," at token_i",token_i

        
    #    level = 0 # tracks how far along we are. gets increased when find a token?
        
           
        def descend_node(curr_node, level):
            
            #print "descending ",curr_node," at level ",level, self.DG.node[curr_node]
                        
            #edges are sorted by weight
            #for e in self.DG.edges(curr_node, data=True):
             #   print "  edge here",e 
            for edge in sorted(list(self.DG.edges(curr_node, data=True)), key=lambda x:x[2]['weight'] ):
                #print 'at edge', edge, 
                #if token_i + level < len(tokens): 
                    #print ' looking for ', tokens[token_i+level]
                #else:
                    #print '  at end of string'
                next_node = edge[1]
                
                #print edge[2]

                if edge[2].get('rule'): # if the edge has a rule
                    
                    if self.match_rule(edge[2]['rule'], tokens, token_i,level)==False:
                        #print 'FAILED RULE'         
                        continue # skip it
                #i#    else:
                    
                        #print 'passed edge rule'#return edge[2]['rule']# return rule here#"MADE IT!",edge[1],edge[2]
                  #  ParserRule = namedtuple('ParserRule', ['production','prev_class','prev_tokens','tokens','next_tokens','next_class'])
       
                if self.DG.node[next_node].get('rule'): # matched nodes have found and rule
                    #print 'found what looking for'
                    return self.DG.node[next_node].get('rule')

                #    return 'found a rule'

                #print tokens, token_i, level, tokens[token_i]
                #print self.DG.node[next_node].get('token')       
                if token_i+level < len(tokens): # do not descend if at end of road
                        

                    if self.DG.node[next_node].get('token')==tokens[token_i+level]:
                
                        #print 'Found successor ',next_node, self.DG.node[next_node]
                        d =descend_node(next_node, level+1)
                        
                        if d:
                            return d
            
    # no matches
        
        return descend_node(0,0)        

    def parse(self,string):
        t_i = 0
        tkns = self.tokenize(string)
        output = ''
        matches = []
        while t_i < len(tkns):
            #print "IN PARSE AT ",t_i, " of ",tkns
            m = self.match_first_at(tkns, t_i)
            #:w
            #if m == None:
                #print "error, no match  at ",t_i," in ",tkns
                #print "tkns[t_i} is", tkns[t_i]
                #print "string = ", string
                #print "matches = ",matches 
            if m==None:
                print "error in string",string,len(string)
            assert m != None # for now, croak on error
            matches.append(m)
            if self.onmatch_rules:
#                            pdb.set_trace()  
                mtkns = [' ']+tkns+[' ']
                mt_i = t_i+1

                for mr in self.onmatch_rules:
#                        pdb.set_trace()
                    (classes,p)=mr
                    (l_c,r_c)=classes
                    # try left match
                    if mt_i < len(l_c) or mt_i+len(r_c)>len(mtkns):
                        continue
#ln                                pdb.set_trace()
        
#                                my_range = range(t_i-len(l_c),t_i+(r_c))
        
                    classes_to_test_l = [self.tokens[c] for c in mtkns[mt_i-len(l_c):mt_i]]
        
                    classes_to_test_r = [self.tokens[c] for c in mtkns[mt_i:mt_i+len(r_c)]]
        
                    if not all(l_c[i] in classes_to_test_l[i] for i in range(len(l_c))):
                        continue
                    if not all(r_c[i] in classes_to_test_r[i] for i in range(len(r_c))):
                        continue
#                             print 'found match rule!!!'
#                                pdb.set_trace()                                

#                    print 'adding'
                    output += p
                    break # break out of for loop
                                
            
            output+=m.production
            t_i += len(m.tokens)
        return ParserOutput(output=output,matches=matches)
    
    def match_all_at(self,tokens, token_i):
        #print "trying tokens ",tokens," at token_i",token_i

        
    #    level = 0 # tracks how far along we are. gets increased when find a token?
        
        matches = []

        def descend_node(curr_node, level):
            
            #print "descending ",curr_node," at level ",level
            
            #edges are sorted by weight
            for edge in sorted(list(self.DG.edges(curr_node, data=True)), key=lambda x:x[2]['weight'] ):
                
                #print 'at edge', edge
                
                next_node = edge[1]
                
                #print edge[2]

                if edge[2].get('rule'): # if the edge has a rule
                    
                    if self.match_rule(edge[2]['rule'], tokens, token_i,level)==False:
                        continue
                    #else:
                    #    print 'passed edge rule'#return edge[2]['rule']# return rule here#"MADE IT!",edge[1],edge[2]
                  #  ParserRule = namedtuple('ParserRule', ['production','prev_class','prev_tokens','tokens','next_tokens','next_class'])
       
                if self.DG.node[next_node].get('rule'): # matched nodes have found and rule
                            matches.append(self.DG.node[next_node].get('rule'))
                            continue
                #    return 'found a rule'

                #else:
                    #print tokens, token_i, level
                
                if token_i+level < len(tokens):
                        
                        if self.DG.node[next_node].get('token')==tokens[token_i+level]:
                
                            #print 'Found successor ',next_node, self.DG.node[next_node]
                            
                            descend_node(next_node, level+1)
            
    # no matches
        
        descend_node(0,0)        
        return matches
    
#def draw_parser_graph(g):
#    pos=nx.spring_layout(g)
#    import matplotlib.pyplot as plt 
#    
#    plt.figure(figsize=(15,15))
#
#    labels = {}
#    for n,d in g.nodes(data=True):
#        label = ''
#        if d.get('token'):
#            label = d['token']
#        elif d.get('found'):
#            label = d['found']
#        labels[n] = label
#    colors = {}
#    
#        for n,d in g.nodes(data=True):
#            color=''
#            if d.get('token'):
#                color = '#A0CBE2'
#            elif d.get('found'):
#                color = '#E2CBA0'
#            colors[n] = color
            
    #    labels=dict((n,d['token']) for n,d in g.nodes(data=True)) # need to change labels for 0,1,etc.

        
#    nx.draw(g,labels=labels,node_color='#2CBACB',node_size=400)

def draw_parser_graph(g):
    num_nodes = len(g.nodes())
    colors = [None]*num_nodes
    labels = {}#[None]*num_nodes
    node_sizes = [None]*num_nodes
    node_shapes = {}#['d']*num_nodes
    len(g.nodes())
    #for n,d in g.nodes(data=True)
    len(colors)

    for n,d in g.nodes(data=True):
        node_shapes[n] = 'd'
        if n==0:
            colors[n] = '#00FF00'
            labels[n] = 'Start'
            node_sizes[n] = 400

        elif d.get('token'):
            colors[n] = '#BA2CCB'
            labels[n] = d['token']
            node_sizes[n] = 600
        else:
 # something wrong here
            colors[n] = '#2CBAFB'
            if not 'found' in d:    
                labels[n]=""#"***********"
            else:# continue
                labels[n] = d['found']
            node_sizes[n] = 300
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,15))


    nx.draw(g,#node_size=node_sizes,#[G.population[v] for v in H],
            node_color=colors,node_size=node_sizes,node_shape='o',#node_shapes,
            edge_color='#A0A0A0',font_size=10,
            labels=labels)

if __name__ == '__main__':
 #   import pdb
 #   pdb.set_trace()
    #nagarip=GraphParser('settings/devanagari.yaml')
    #print nagarip.parse(" kyaa hu)aa hai bhaa))ii")
    urdup = GraphParser('settings/urdu.yaml')
#    pdb.set_trace()
    x=urdup.parse('haa;n jii')
    print x.output

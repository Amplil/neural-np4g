import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
#import slacknotice # オリジナルモジュール slacknotice.send("")
import traceback
import random

# 入力ノード数エラー：-1
# 出力およびその他のエラー：-2

class p(): # ONのときだけ表示
    out=False
    @classmethod
    def on(cls):
        cls.out=True
    @classmethod
    def off(cls):
        cls.out=False
    @classmethod
    def rint(cls,*args): # p.rint()関数
        if cls.out:
            print(*args)

def sigmoid(gn): # シグモイド関数
    # sumを関数内で行う必要があるのかどうか今後検討
    x=np.sum(gn.in_val()) # w1*in1+w2*in2+...+wt*int
    result = 1 / (1 + np.exp(-x)) # シグモイド関数の演算
    #for out_node in gn.out_nodes:  # Output to all connected nodes
    #    gn.out_ele(out_node, result)
    return result

def tanh(gn): # tanh関数
    # sumを関数内で行う必要があるのかどうか今後検討
    x=np.sum(gn.in_val()) # w1*in1+w2*in2+...+wt*int
    result = np.tanh(x)
    return result

def out_(gn): # 出力関数
    # sumを関数内で行う必要があるのかどうか今後検討
    result=np.sum(gn.in_val()) # w1*in1+w2*in2+...+wt*int
    return result

def derivative(gn): # 導関数
    #x=np.sum(gn.in_vals) # w1*in1+w2*in2+...+wt*int
    if gn.attr==tanh: # ノードの関数がtanhの場合
        return 1-gn.val()**2 # d/dx(tanh(x))=1-y^2
    elif gn.attr==out_: # ノードの関数がout_の場合
        return 1 # d/dx(x)=1


class GraphNode(): # グラフ（ネットワーク）の1つのノードにフォーカス
    def __init__(self,G,name):
        self.G=G
        self.name=name
        if (name in G.nodes)==False:
            p.rint('Error: ',name,' does not exist in the graph')
            return None
        else:
            self.in_deg=G.in_degree(name)
            self.out_deg=G.out_degree(name)
            self.in_nodes=list(G.pred[name])
            self.out_nodes=list(G[name])

            self.ref=G.nodes[name] # self.ref['attribute'],self.ref['value']で参照ができる
            self.in_refs=[G.edges[n,name] for n in self.in_nodes] # self.in_refs[i]['value']で参照できる
            self.out_refs=[G.edges[name,n] for n in self.out_nodes] # in self.out_refs[i]['value']で参照できる

            self.attr=self.ref['attribute'] # attrは定数扱いとするため変数関数にしない
            # 変数関数
            self.val=lambda val=None: self.node_ref('value',val)
            self.d=lambda d=None: self.node_ref('delta',d)
            self.in_val=lambda val=None: self.edge_ref('value',True,val)
            self.out_val=lambda val=None: self.edge_ref('value',False,val)
            self.in_w=lambda w=None: self.edge_ref('weight',True,w)
            self.out_w=lambda w=None: self.edge_ref('weight',False,w)
            self.in_d=lambda d=None: self.edge_ref('delta',True,d)
            self.out_d=lambda d=None: self.edge_ref('delta',False,d)
        
    def node_ref(self,name='value',val=None): # ノードを参照する
        if val is None: # val()で値を出力
            return self.ref[name]
        else:
            self.ref[name]=np.float64(val)
            return np.float64(val)
    def edge_ref(self,name='value',inflow=True,val=None): # エッジを参照する、流出の場合inflow=False
        if inflow:
            refs=self.in_refs
            deg=self.in_deg
        else:
            refs=self.out_refs
            deg=self.out_deg
        if val is not None:
            if type(val)==list or type(val)==np.ndarray:
                if len(val)==deg: # 代入する値の個数がノードの個数と合っているかどうか
                    for i,ref in enumerate(refs):
                        ref[name]=np.float64(val[i])
                    return np.array(val)
                else:
                    print("代入する値の個数がノードの個数と合いません。")
                    return False
            else: # リストでなければ数値が考えられる。それ以外はエラー
                val=np.float64(val)
                for ref in refs:
                    ref[name]=np.float64(val) # nameを参照して値を変える
                return np.float64(val)
        else: # valが指定されてないとき
            return np.array([ref[name] for ref in refs]) # ノードの値を出力する
        

class Nodes(): # 複数のノードそれぞれにフォーカス
    def __init__(self,G,attr=""):
        self.G=G
        #self.names=list(G.nodes)
        self.all_gns=[GraphNode(G,name) for name in list(G.nodes)] # すべてのノードのGraphNodeのリスト
        if attr=="callable": # 入力ノード以外の呼び出し可能な属性だけのGraphNodeのリスト
            self.gns=[gn for gn in self.all_gns if callable(gn.attr)]
        elif attr!="": # 属性が指定されてる場合はその属性だけのGraphNodeのリスト
            self.gns=[gn for gn in self.all_gns if gn.attr==attr]
        else: # デフォルトですべてのノードを選択
            self.gns=self.all_gns

    def val(self,val=None): # 複数ノードの値の出力と更新
        if val is not None:
            if type(val)==list or type(val)==np.ndarray:
                if len(val)==len(self.gns): # 代入する値の個数がノードの個数と合っているかどうか
                    for i,gn in enumerate(self.gns):
                        gn.val(np.float64(val[i])) # valを参照して値を変える
                    return np.array(val)
                else:
                    print("代入する値の個数がノードの個数と合いません。")
                    return False
            else: # リストでなければ数値が考えられる。それ以外はエラー
                for gn in self.gns:
                    gn.val(np.float64(val)) # valを参照して値を変える
                return np.float64(val)
        else: # valが指定されてないとき
            return np.array([gn.val() for gn in self.gns]) # ノードの値を出力する
        
    def d(self,d=None): # 複数ノードの値の出力と更新
        if d is not None:
            if type(d)==list or type(d)==np.ndarray:
                if len(d)==len(self.gns): # 代入する値の個数がノードの個数と合っているかどうか
                    for i,gn in enumerate(self.gns):
                        gn.d(np.float64(d[i])) # valを参照して値を変える
                    return np.array(d)
                else:
                    print("代入する値の個数がノードの個数と合いません。")
                    return False
            else: # リストでなければ数値が考えられる。それ以外はエラー
                for gn in self.gns:
                    gn.d(np.float64(d)) # valを参照して値を変える
                return np.float64(d)
        else: # valが指定されてないとき
            return np.array([gn.d() for gn in self.gns]) # ノードの値を出力する
        
    def edges_out(self): # すべてのノードの値を出力エッジに反映させる
        for gn in self.gns:
            gn.out_val(gn.val()*gn.out_w()) # ノードの値を出力エッジに反映させる
        
    def edges_in(self): # すべてのノードの誤差deltaを入力エッジに反映させる
        for gn in self.gns:
            gn.in_d(gn.d()) # ノードの値を出力エッジに反映させる

    def edges_direct_out(self): # すべてのノードの値を出力エッジに反映させる
        for gn in self.gns:
            gn.out_val(gn.val()) # ノードの値を出力エッジに反映させる

    def select(self,attr=""): # 属性を指定してノードを抽出する
        if attr=="callable": # 入力ノード以外の呼び出し可能な属性だけのGraphNodeのリスト
            self.gns=[gn for gn in self.all_gns if callable(gn.attr)]
        elif attr!="": # 属性が指定されてる場合はその属性だけのGraphNodeのリスト
            self.gns=[gn for gn in self.all_gns if gn.attr==attr]
        else: # デフォルトですべてのノードを選択
            self.gns=self.all_gns
        return self


class NetworkProgram(): # ネットワーク構造データからプログラムを実行
    #def __init__(self,input,node_struct,edge_struct):
    def __init__(self,node_struct,edge_struct):
        self.input=input
        self.G=nx.DiGraph()
        self.states=[]

        self.G.add_nodes_from([(tup[0],{'attribute':tup[1],'value': np.float64(0),'delta': np.float64(0)}) for tup in node_struct]) # node_structをnetworkxに対応した形にして渡す
        self.G.add_edges_from(edge_struct)
        self.G.add_edges_from(list(map(lambda tup: tup+({'value': np.float64(0),'delta': np.float64(0)},) ,self.G.edges))) # エッジ要素を0.で初期化する
        self.G.add_weighted_edges_from(list(map(lambda tup: tup+(np.random.rand(),) ,self.G.edges))) # 重みを乱数で初期化する

    def view_network(self):
        p.rint("nodes: ",self.G.nodes.data()) # 必要に応じて表示
        p.rint("edges: ",self.G.edges.data())
        return nx.nx_agraph.view_pygraphviz(self.G,prog='dot')  # pygraphvizが必要

    def network_info(self):
        print("nodes: ",self.G.nodes.data())
        print("edges: ",self.G.edges.data())

    def run_tick(self,nodes):
        #self.states.append(Nodes(self.G).value()) # 状態の保存（すべてのノードの値）
        nodes().edges_out() # 先にすべてのノードの値を出力エッジに反映させる

        for gn in nodes("callable").gns: # （入力ノード以外の）実行のできるノード
            gn.val(gn.attr(gn)) # ノードの実行してノードの値を変える
            p.rint("result node:",gn.name,", value:",gn.val())
        output=nodes(out_).val() # 出力ノードの値
        self.states.append(nodes().val()) # 状態の保存（すべてのノードの値）
        p.rint("output: ",output)
        return output

    def forward(self,inputs_list):
        outputs=[]
        self.states=[]
        #self.out_nodes=out_nodes
        nodes=Nodes(self.G).select # selectありきでnodesを組む。つまりnodesは関数
        nodes().val(0) # すべてのノードの値を0にリセットする
        nodes().edges_out() # 先にすべてのノードの値を出力エッジに反映させる
        #in_nodes=Nodes(self.G,"in") # すべての入力ノード
        for inputs in inputs_list: # inputs_listの長さ分実行
            p.rint("inputs: ",inputs)
            nodes("in").val(inputs) # 入力ノードにinputsの値をそれぞれ入れていく
            outputs.append(self.run_tick(nodes))
        nodes("in").val(0) # 入力し終わったら、すべての入力ノードの値を0にリセットする

        N=1 # 繰り返し回数
        for _ in range(N): # 入力が終わったあとN回繰り返す
            #self.run_tick(inputs[i] if len(inputs)>i else []) # inputが存在するまで
            #output=self.run_tick()
            #output=Nodes(self.G,out_).value() # 出力ノードの値をoutputに書き出す
            #output=[gn.val for gn in self.nodes if gn.attr==out_] # 出力ノードの値をoutputに書き出す
            outputs.append(self.run_tick(nodes))
            #p.rint("output: ",output)
        return np.array(outputs)
    
    def backward(self,outputs_list):
        out_reverse=np.array(outputs_list)[::-1] # 逆順にする
        state_reverse=self.states[::-1] # 逆順にする
        self.eta=1 # 暫定値
        nodes=Nodes(self.G).select # selectありきでnodesを組む。つまりnodesは関数
        nodes().d(0) # 誤差deltaのリセット
        nodes().edges_in() # すべてのノードの誤差deltaを入力エッジに反映させる
        for i,outputs in enumerate(out_reverse):
            p.rint("train data: ",outputs)
            nodes().val(state_reverse[i]) # 状態を読みこむ
            nodes().edges_direct_out() # 先にすべてのノードの値を出力エッジに反映させる

            #out_nodes=Nodes(self.G,out_) # すべての出力ノード
            ds=nodes(out_).val()-outputs # 出力ノードの誤差
            nodes(out_).d(ds) # 出力ノードの誤差を入れていく

            for gn in nodes("callable").gns: # （入力ノード以外の）実行のできるノード
                if gn.attr!=out_: # 出力ノード以外->出力エッジがなければ実行されないためいらない
                    #gn.ref['delta']+=derivative(gn)*np.sum(gn.out_ds*gn.out_ws) # 誤差deltaを加える（出力誤差との足し合わせ）
                    gn.d(derivative(gn)*np.sum(gn.out_d()*gn.out_w())) # 誤差deltaを加える（出力誤差との足し合わせ）

                dw=-self.eta*gn.d()*gn.in_val() # 重みの更新量
                #dw=self.eta*gn.ref['delta']*in_val # 重みの更新量
                #gn.in_ws[j]+=dw # dwだけ更新する
                #gn.in_refs[j]['weight']+=dw # dwだけ更新する
                gn.in_w(gn.in_w()+dw) # dwだけ加えて更新する
                p.rint("update node ",gn.name,", delta:",gn.d(),", weights:",gn.in_w())

            nodes().edges_in() # すべてのノードの誤差deltaを入力エッジに反映させる
            #self.network_info()

        nodes().val(0) # 状態を読みこむ
        nodes().edges_out() # 先にすべてのノードの値を出力エッジに反映させる
        return self.network_info()


def adfs(gn,node_body,edge_struct): # 自動定義関数
    #if (gn.in_deg==1 and gn.out_deg==1): # 入力も出力もノードは1つ
    p.rint("adfs node_body: ",node_body)
    p.rint("adfs edge_struct: ",edge_struct)
    if (gn.in_deg==1): # 入力は1つ
        if (type(gn.in_ele_list[0])!=list): # 繰り返し処理に対応
            in_list=[gn.in_ele_list[0]]
        else:
            in_list=gn.in_ele_list[0]

        out_list=[]
        #p.rint("adfs input:",in_list)
        for input in in_list: # 繰り返し処理に対応
            p.rint("adfs input:",input)
            #node_struct=[('S',input)]
            #node_struct.extend(node_body)
            #node_struct.append(('out',out_func))
            #gsp=NetworkProgram(input,node_struct,edge_struct)
            gsp=NetworkProgram(input,node_body,edge_struct)
            out=gsp.run()
            out_list.append(out)
        if (len(out_list)==1):
            result=out_list[0]
        else:
            result=out_list
        #return gn.out_ele(gn.out_node_list[0],result) # resultを代入、出力ノードは1つだけ

        p.rint(result)
        for out_node in gn.out_node_list: # 出力ノードは複数でも可
            gn.out_ele(out_node,result)
        p.rint("adfs out:",result)
        return result
    else: return -1


class NP4Gstruct(): # ネットワーク生成オブジェクト
    def __init__(self,num_nodes,*funcs):
        self.num_nodes=num_nodes
        self.repeat_num=0 # 繰り返し回数
        self.clear1=0
        #self.node_body=[]
        #self.edge_struct=[]
        #self.adfs_list=[div,sum,equal,control_gate,control_not_gate,pos,object_func] # まずは初期関数リストでadfsリストを定義 出力関数は入れない
        self.adfs_list=[]
        self.add_adfs(*funcs) # ネットワーク生成に使う関数を登録する

    def add_adfs(self,*add_tupple): # *add_tuppleは可変長引数
        for node in add_tupple:
            if not (node in self.adfs_list): # すでにadfsに登録されているものは追加しない
                self.adfs_list.append(node) # （入出力）オブジェクト(文字列)をadfsリストに登録

    def random_struct(self): # node_num: ノードを何個とるか inを含めない
        #node_num=3 # ノードを何個とるか inを含めない
        #node_list=[input]
        node_list=[] # inを含めない
        #node_list.extend([random.choice(adfs_list) for n in range(node_num)]) # ランダムに選んで加える
        node_list.extend(random.choices(self.adfs_list,k=self.num_nodes)) # ランダムに選んで加える
        node_struct=[('S','dummy')] # あとでinputは抜かす
        node_struct.extend(list(enumerate(node_list))) # inも含めたランダムに作られたノード構造
        node_name_list=[i for i,_ in node_struct] # node_nameだけのリスト
        #self.node_name_list=node_name_list
        if not (True in map(callable,[i for _,i in node_struct])): # node_structに呼び出し可能なノード(関数)が1つもない場合
            print("random_struct: All nodes are objects: skip")
            node_body=[]
            edge_struct=[]
            return node_body,edge_struct # すべてオブジェクトノードであった場合、inをedge_structに入れることができなくなってしまう。

        #connect_list=[]
        edge_struct=[]
        n=0
        while not 'S' in [i for i,_ in edge_struct]: # inを含むedge_structができるまでランダムに作り続ける
            n+=1
            if(n>1000):
                print("random_struct: ",n,"回edgeの構築を繰り返しましたが、inを含むedge_structができませんでした。node_struct: ",node_struct)
                node_body=[]
                edge_struct=[]
                return node_body,edge_struct
            edge_struct=[]
            #for i,node in enumerate(node_list):
            for node_name,node_content in node_struct:
                if callable(node_content):
                    node_sample=list(filter(lambda x: x!=node_name,node_name_list)) # 当ノードを除いたサンプルを用意する
                    #node_sample.remove(node_name) # 当ノードを除いたサンプルを用意する

                    #print(node_content)
                    #print(node_sample)
                    #print(node_name_list)
                    ncn=node_content.__name__
                    if ncn=="equal" or ncn=="control_gate" or ncn=="control_not_gate":
                        #connect_list.extend([i,i]) # 2つ取る
                        edge_struct.extend([(in_node,node_name) for in_node in random.sample(node_sample,2)]) # 入力ノードを重複なしでランダムに2つ選ぶ
                    elif ncn=="sum":
                        repeat=random.randrange(1,self.num_nodes) # sumの入力の本数は1～node_num（取れる最大のノード数）のうちでランダムに決める
                        edge_struct.extend([(in_node,node_name) for in_node in random.sample(node_sample,repeat)]) # 入力ノードを重複なしでランダムにrepeat分選ぶ
                        #connect_list.extend([i for _ in range(repeat)]) # repeat分取る
                    else:
                        edge_struct.append((random.choice(node_sample),node_name)) # 入力ノードをランダムに1つ選ぶ
                        #connect_list.append(i) # 1つ取る
                        #print('else')
            """
            if not 'S' in [i for i,_ in edge_struct] :
                print(node_struct)
                print(edge_struct)
                print('no in')
            """
        """
        edge_struct=[]
        for connect_num in connect_list:
            node_num_range=list(range(node_num+1)) # inも含める
            node_num_range.pop(connect_num) # 当connectは含めない
            edge_struct.append((random.choice(node_num_range),connect_num))
        """
        node_body=node_struct[1:] # inputを抜かす
        return node_body,edge_struct # node_body,edge_structを渡す

    def Search2RequirementsWithAnalysis(self,input1,out_expect1,input2,out_expect2,timelimit=0,interval=0): # 2条件での解析を伴う探索
        self.input1=input1
        self.out_expect1=out_expect1
        self.input2=input2
        self.out_expect2=out_expect2
        self.start_time=time.time()
        self.half_time=time.time()
        self.timelimit=timelimit
        self.interval=interval
        self.result1=""
        self.result2=""
        #self.adfs_list.extend([self.input1,self.out_expect1,self.input2,self.out_expect2]) # （入出力）オブジェクト(文字列)をadfsリストに登録
        #add_list=[self.input1,self.out_expect1,self.input2,self.out_expect2]
        """
        for node in add_list:
            if not (node in self.adfs_list): # すでにadfsに登録されているものは追加しない
                self.adfs_list.append(node) # （入出力）オブジェクト(文字列)をadfsリストに登録
        """
        self.add_adfs(self.input1,self.out_expect1,self.input2,self.out_expect2)
        n=0 # インターバル内の繰り返し回数
        while self.result1!=self.out_expect1 or self.result2!=self.out_expect2:
            self.repeat_num+=1
            n+=1
            if n>1000:
                n=0
                self.now_time=time.time()
                elapsed_time=self.now_time-self.start_time
                if self.timelimit>0 and elapsed_time>=self.timelimit:
                    print(elapsed_time,"秒経過しましたが、条件を満たすネットワークを見つけることができませんでした。")
                    #slacknotice.send(elapsed_time,"秒経過しましたが、条件を満たすネットワークを見つけることができませんでした。")
                    return -1

                if self.interval>0 and self.now_time-self.half_time>=self.interval:
                    print(elapsed_time,"秒経過 ","生成されたグラフの数: ",self.repeat_num,",第一条件をクリアしたグラフの数: ",self.clear1)
                    #slacknotice.send(elapsed_time,"秒経過 ","生成されたグラフの数: ",self.repeat_num,",第一条件をクリアしたグラフの数: ",self.clear1)
                    self.half_time=self.now_time
                    self.clear1=0
                    self.repeat_num=0
            node_body,edge_struct=self.random_struct()
            #print(node_body)
            #print(edge_struct)
            self.gsp1=NetworkProgram(self.input1,node_body,edge_struct)
            self.result1=self.gsp1.run()
            if self.result1!=self.out_expect1:
                continue
            #print("result1 is ok")
            self.clear1+=1
            self.gsp2=NetworkProgram(self.input2,node_body,edge_struct)
            self.result2=self.gsp2.run()
        self.add_adfs(lambda gn : adfs(gn,node_body,edge_struct)) # 条件を満たすネットワークをadfsリストに登録
        self.gsp1.network_show()
        #self.gsp2.network_show()
        return node_body,edge_struct

    def Search1Requirement(self,input,out_expect): # 1条件での探索
        self.result=""
        self.add_adfs(input)
        while self.result!=out_expect:
            node_body,edge_struct=self.random_struct()
            self.gsp=NetworkProgram(input,node_body,edge_struct)
            self.result=self.gsp.run()
        self.add_adfs(lambda gn : adfs(gn,node_body,edge_struct)) # 条件を満たすネットワークをadfsリストに登録
        self.gsp.network_show()
        return node_body,edge_struct

    def MultiRequirements(self,teacher_data,timelimit=0,interval=0): # 複数条件での探索
        self.start_time=time.time()
        self.half_time=time.time()
        self.timelimit=timelimit
        self.interval=interval
        self.result=""

        for data in teacher_data:
            self.add_adfs(*data)
        n=0 # インターバル内の繰り返し回数
        while True:
            self.repeat_num+=1
            n+=1
            if n>1000:
                n=0
                self.now_time=time.time()
                elapsed_time=self.now_time-self.start_time
                if self.timelimit>0 and elapsed_time>=self.timelimit:
                    print(elapsed_time,"秒経過しましたが、条件を満たすネットワークを見つけることができませんでした。")
                    #slacknotice.send(elapsed_time,"秒経過しましたが、条件を満たすネットワークを見つけることができませんでした。")
                    return [],[]

                if self.interval>0 and self.now_time-self.half_time>=self.interval:
                    print(elapsed_time,"秒経過 ","生成されたグラフの数: ",self.repeat_num,",第一条件をクリアしたグラフの数: ",self.clear1)
                    #slacknotice.send(elapsed_time,"秒経過 ","生成されたグラフの数: ",self.repeat_num,",1つでも条件をクリアしたグラフの数: ",self.clear1)
                    self.half_time=self.now_time
                    self.clear1=0
                    self.repeat_num=0

            node_body,edge_struct=self.random_struct()
            #print(node_body)
            #print(edge_struct)
            #pdb.set_trace()
            for data in teacher_data:
                input=data[0]
                out_expect=data[1]
                self.gsp=NetworkProgram(input,node_body,edge_struct)
                self.result=self.gsp.run()
                if self.result!=out_expect:
                    break # 結果が合わなかったらforからbreak
                else:
                    self.clear1+=1
            else: # forが最後までいったら関数自体を終了
                self.add_adfs(lambda gn : adfs(gn,node_body,edge_struct)) # 条件を満たすネットワークをadfsリストに登録
                print("実行時間：",time.time()-self.start_time,"秒")
                self.gsp.network_show()
                return node_body,edge_struct

    def PhasedGenerate(self,input1,out_expect1,input2,out_expect2,timelimit=0,interval=0):
        #add_list=[input1,out_expect1,input2,out_expect2]
        self.add_adfs(input1,out_expect1,input2,out_expect2)
        self.Search1Requirement(input1,out_expect1)
        self.Search1Requirement(input2,out_expect2)
        return self.Search2RequirementsWithAnalysis(input1,out_expect1,input2,out_expect2,timelimit=timelimit,interval=interval)

    def random_structMulti(self): # node_num: ノードを何個とるか inを含めない
        node_list=[] # inを含めない
        node_list.extend(random.choices(self.adfs_list,k=self.num_nodes)) # ランダムに選んで加える
        node_struct=[('in1','dummy1'),('in2','dummy2')] # あとでin1,in2は抜かす
        node_struct.extend(list(enumerate(node_list))) # inも含めたランダムに作られたノード構造
        node_name_list=[i for i,_ in node_struct] # node_nameだけのリスト
        #self.node_name_list=node_name_list
        if not (True in map(callable,[i for _,i in node_struct])): # node_structに呼び出し可能なノード(関数)が1つもない場合
            print("random_struct: All nodes are objects: skip")
            node_body=[]
            edge_struct=[]
            return node_body,edge_struct # すべてオブジェクトノードであった場合、inをedge_structに入れることができなくなってしまう。

        #connect_list=[]
        edge_struct=[]
        n=0
        while not ('in1' in [i for i,_ in edge_struct] and 'in2' in [i for i,_ in edge_struct]): # in1,in2を含むedge_structができるまでランダムに作り続ける
            n+=1
            if(n>1000):
                print("random_struct: ",n,"回edgeの構築を繰り返しましたが、inを含むedge_structができませんでした。node_struct: ",node_struct)
                node_body=[]
                edge_struct=[]
                return node_body,edge_struct
            edge_struct=[]
            #for i,node in enumerate(node_list):
            for node_name,node_content in node_struct:
                if callable(node_content):
                    node_sample=list(filter(lambda x: x!=node_name,node_name_list)) # 当ノードを除いたサンプルを用意する

                    #print(node_content)
                    #print(node_sample)
                    #print(node_name_list)
                    ncn=node_content.__name__
                    if ncn=="equal" or ncn=="control_gate" or ncn=="control_not_gate":
                        #connect_list.extend([i,i]) # 2つ取る
                        edge_struct.extend([(in_node,node_name) for in_node in random.sample(node_sample,2)]) # 入力ノードを重複なしでランダムに2つ選ぶ
                    elif ncn=="sum":
                        repeat=random.randrange(1,self.num_nodes) # sumの入力の本数は1～node_num（取れる最大のノード数）のうちでランダムに決める
                        edge_struct.extend([(in_node,node_name) for in_node in random.sample(node_sample,repeat)]) # 入力ノードを重複なしでランダムにrepeat分選ぶ
                        #connect_list.extend([i for _ in range(repeat)]) # repeat分取る
                    else:
                        edge_struct.append((random.choice(node_sample),node_name)) # 入力ノードをランダムに1つ選ぶ
                        #connect_list.append(i) # 1つ取る
                        #print('else')
        node_body=node_struct[2:] # inputを抜かす
        return node_body,edge_struct # node_body,edge_structを渡す

    def Search1RequirementMulti(self,in1,in2,out_expect): # 1条件での探索
        #self.result=""
        self.result=[""]
        self.add_adfs(in1,in2,out_expect)
        while self.result!=out_expect:
            node_in12,edge_in12=self.random_struct()
            mi=MultiInout(node_in12,edge_in12)
            #self.result=mi.run(in1,in2)
            self.output=mi.run(in1,in2)
            if type(self.output)!=list or self.output==[]:
                self.result=[""]
            else:
                self.result=self.output[-1] # 最後の出力が第一出力
        self.add_adfs(lambda gn : adfs_in12(gn,node_in12,edge_in12)) # 条件を満たすネットワークをadfsリストに登録
        #self.gsp.network_show()
        return node_in12,edge_in12

    def MultiRequirementsMulti(self,teacher_data,timelimit=0,interval=0): # 複数条件での探索
        # teacher_data: ((in1,in2,out_expect),(in12,in22,out_expect2), ...)
        self.start_time=time.time()
        self.half_time=time.time()
        self.timelimit=timelimit
        self.interval=interval
        self.result=""

        for data in teacher_data:
            self.add_adfs(*data)
        n=0 # インターバル内の繰り返し回数
        while True:
            self.repeat_num+=1
            n+=1
            if n>1000:
                n=0
                self.now_time=time.time()
                elapsed_time=self.now_time-self.start_time
                if self.timelimit>0 and elapsed_time>=self.timelimit:
                    print(elapsed_time,"秒経過しましたが、条件を満たすネットワークを見つけることができませんでした。")
                    #slacknotice.send(elapsed_time,"秒経過しましたが、条件を満たすネットワークを見つけることができませんでした。")
                    return [],[]

                if self.interval>0 and self.now_time-self.half_time>=self.interval:
                    print(elapsed_time,"秒経過 ","生成されたグラフの数: ",self.repeat_num,",第一条件をクリアしたグラフの数: ",self.clear1)
                    #slacknotice.send(elapsed_time,"秒経過 ","生成されたグラフの数: ",self.repeat_num,",1つでも条件をクリアしたグラフの数: ",self.clear1)
                    self.half_time=self.now_time
                    self.clear1=0
                    self.repeat_num=0

            node_body,edge_struct=self.random_structMulti()
            #print(node_body)
            #print(edge_struct)
            #pdb.set_trace()
            for data in teacher_data:
                in1=data[0]
                in2=data[1]
                out_expect=data[2]
                mi=MultiInout(node_body,edge_struct)
                self.output=mi.run(in1,in2)
                if type(self.output)!=list or self.output==[]:
                    self.result=[""]
                else:
                    self.result=self.output[-1] # 最後の出力が第一出力

                if self.result!=out_expect:
                    break # 結果が合わなかったらforからbreak
                else:
                    self.clear1+=1
            else: # forが最後までいったら関数自体を終了
                self.add_adfs(lambda gn : adfs_in12(gn,node_body,edge_struct)) # 条件を満たすネットワークをadfsリストに登録
                print("実行時間：",time.time()-self.start_time,"秒")
                #self.gsp.network_show()
                return node_body,edge_struct    



# 出力されたネットワーク情報を利用可能な形に変換
class TransNetworkInfo():
    node_dict={}
    def __init__(self,node_info_str,edge_info_str):
        self.node_info_str=node_info_str
        self.edge_info_str=edge_info_str
    @classmethod
    def node_resist(cls,node_key,node_str):
        cls.node_dict[node_key]=node_str

    def trans_info_raw(self): # もともとに近いネットワーク情報
        #return self.trans_node_info(),self.trans_edge_info()
        print("node: ",self.trans_node_raw())
        print("edge: ",self.trans_edge())
    def trans_info(self): # 見やすくしたネットワーク情報
        #return self.trans_node_info(),self.trans_edge_info()
        print("node: ",self.trans_node_str_list())
        print("edge: ",self.trans_edge())
    def trans_edge(self):
        edge_info=eval(self.edge_info_str)
        return [(edge[0],edge[1]) for edge in edge_info]
    def trans_node_raw(self):
        node_info=self.node_info_str
        for key in self.node_dict.keys():
            node_info=node_info.replace(key,"'"+self.node_dict[key]+"'")
        node_info=eval(node_info.replace(" <"," '<").replace(">}",">'}").replace(">,",">',").replace(">)",">')"))
        #return [(node[0], node[1]['attribute'].split()[1] if (type(node[1]['attribute'])==str and "function" in node[1]['attribute']) else node[1]['attribute'] ) for node in node_info]
        if 'attribute' in node_info[0][1]:
            return [(node[0],node[1]['attribute']) for node in node_info]
        else:
            return node_info
    def trans_node_str_list(self):
        node_info=self.trans_node_raw()
        #return [(node[0], node[1].split()[1] if (type(node[1])==str and "function" in node[1]) else node[1] ) for node in node_info]
        node_list=[]
        for node in node_info:
            """
            if node[1] in self.node_dict:
                node_list.append((node[0],self.node_dict[node[1]]))
            """
            if type(node[1])==str and ("function" in node[1]):
                node_list.append((node[0], node[1].split()[1]))
            elif node[0]=='S' or node[0]=='out':
                continue
            else:
                node_list.append((node[0], node[1]))
        return node_list
    def trans_node_body(self):
        node_info=self.trans_node_str_list()
        #return [(node[0], (eval(node[1]) if callable(eval(node[1])) else node[1])) for node in node_info]
        node_body=[]
        for node in node_info:
            try:
                node1=eval(node[1])
                if not callable(node1):
                    node1=node[1]
            except:
                node1=node[1]
            node_body.append((node[0], node1))
        return node_body

    def trans_node_body_str(self):
        node_str='['
        for node in self.trans_node_str_list():
            if type(node[0])!=str:
                node_str+='('+str(node[0])+','
            else:
                node_str+='(\''+str(node[0])+'\','

            try:
                if callable(eval(node[1])):
                    node_str+=str(node[1])+'),'
                else:
                    node_str+='\''+str(node[1])+'\'),'
            except:
                node_str+='\''+str(node[1])+'\'),'
        return node_str[:-1]+']'

    def edge_node(self): # NetworkProgramオブジェクトとして利用可能なノードとエッジのリスト
        return [self.trans_node_body(),self.trans_edge()]

    def mermaid(self): # Mermaid記法で表示
        print('flowchart LR;\nS([S])')
        for node in self.trans_node_str_list():
            if node[1]=='sum':
                node1='[+]'
            elif node[1]=='equal':
                node1='[==]'
            elif node[1]=='control_gate':
                node1='( )'
            elif node[1]=='control_not_gate':
                node1='("×")'
            elif node[1]=='control_not_gate':
                node1='("×")'
            else: node1='(["'+node[1]+'"])'
            print(str(node[0])+node1)
        for edge in self.trans_edge():
            print(edge[0],'-->',edge[1])


def NP4Gtest(node_body,edge_struct,test_data):
    result=True
    for in_out in test_data:
        nptest=NetworkProgram(in_out[0],node_body,edge_struct)
        ans=nptest.run()
        correct=in_out[1]==ans
        print("input:",in_out[0],", answer:",ans,", correct:",correct)
        if not correct:
            result=False
    return result

class MultiInout(): # 複数入出力をサポートするNetworkProgram
    def __init__(self,node_body_in12,edge_struct_in12):
        self.node_body_in12=node_body_in12
        self.edge_struct_in12=edge_struct_in12
    def run(self,in1,in2): # 複数入出力用のrunメソッド
        node_body=self.node_body_in12.copy()
        edge_struct=self.edge_struct_in12.copy()
        node_body.append(('in2',in2)) # in2をnode_bodyに追加
        for i in range(len(edge_struct)): # in1をSに置き換え
            if edge_struct[i][0]=='in1':
                edge_struct[i]=('S',edge_struct[i][1])
        self.np=NetworkProgram(in1,node_body,edge_struct)
        self.np.run()
        for i,output in enumerate(reversed(self.np.output)): # outputはリスト
            p.out("out",i+1,": ",output)
        return self.np.output # outputはリスト
    def test(self,multi_test_data): # multi_test_dataのoutはまだリストでない
        self.result=True
        for in_out in multi_test_data:
            ans=self.run(in_out[0],in_out[1])[-1] # 最後のoutを取得
            correct=in_out[2]==ans
            print("in1:",in_out[0],", in2:",in_out[1],", answer:",ans,", correct:",correct)
            if not correct:
                self.result=False
        return self.result

def adfs_in12(gn,node_body_in12,edge_struct_in12): # 2入力の自動定義関数
    #if (gn.in_deg==1 and gn.out_deg==1): # 入力も出力もノードは1つ
    p.out("adfs_in12 node_body_in12: ",node_body_in12)
    p.out("adfs_in12 edge_struct_in12: ",edge_struct_in12)
    if (gn.in_deg==2): # 入力は2つ
        if (type(gn.in_ele_list[0])!=list and type(gn.in_ele_list[1])!=list): # 2つの入力のどちらもリストでないとき
            attr=[[gn.in_ele_list[i]] for i in range(2)] # 個々にリスト化
        elif (type(gn.in_ele_list[0])==list and type(gn.in_ele_list[1])==list and len(gn.in_ele_list[0])==len(gn.in_ele_list[1])): # 入力がどちらもリストであり、要素数も同じとき
            attr=gn.in_ele_list
        else:
            p.out("adfs_in12 nodes list error")
            return -2
        out_list=[]
        #p.out("adfs input:",in_list)
        for i in range(len(attr[0])): # 繰り返し処理に対応
            p.out("adfs_in12 in1:",attr[0][i],", in2:",attr[1][i])
            mi=MultiInout(node_body_in12,edge_struct_in12)
            out=mi.run(attr[0][i],attr[1][i])[-1] # 今の段階では1出力を想定する
            out_list.append(out)
        if (len(out_list)==1):
            result=out_list[0]
        else:
            result=out_list

        p.out(result)
        for out_node in gn.out_node_list: # 出力ノードは複数でも可
            gn.out_ele(out_node,result)
        p.out("adfs_in12 out:",result)
        return result
    else: return -1

if __name__ == "__main__":
    """
    np1=NetworkProgram([('x0','in'),('h0',sig),('y0',out_)],[('x0','h0'),('h0','y0')])
    p.on()
    np1.forward([[1]])
    """
    node3=[('x0','in'),('x1','in'),('x2','in'),
        ('h0',tanh),('h1',tanh),('h2',tanh),
        ('y0',out_),('y1',out_),('y2',out_)]
    edge3=[('x0','h0'),('x0','h1'),('x0','h2'),
        ('x1','h0'),('x1','h1'),('x1','h2'),
        ('x2','h0'),('x2','h1'),('x2','h2'),
        ('h0','y0'),('h0','y1'),('h0','y2'),
        ('h1','y0'),('h1','y1'),('h1','y2'),
        ('h2','y0'),('h2','y1'),('h2','y2')]
    np3=NetworkProgram(node3,edge3)

    # 重みを指定
    np3.G.add_weighted_edges_from(
        [('x0','h0',0),('x0','h1',1),('x0','h2',0),
        ('x1','h0',1),('x1','h1',0),('x1','h2',0),
        ('x2','h0',0),('x2','h1',0),('x2','h2',0),
        ('h0','y0',1),('h0','y1',0),('h0','y2',0),
        ('h1','y0',0),('h1','y1',1),('h1','y2',0),
        ('h2','y0',0),('h2','y1',0),('h2','y2',1)])

    p.on()
    inputs_list=[[1,0,0],[0,1,0]]
    outputs_list=[[0,0,0],[0,1,0],[1,0,0]]
    np3.forward(inputs_list)
    np3.backward(outputs_list)
    #np3.forward(inputs_list)
    #np3.backward(outputs_list)

    """
    node_rnn=[('x0','in'),('x1','in'),('x2','in'),
            ('h0',tanh),('h1',tanh),('h2',tanh),
            ('y0',out_),('y1',out_),('y2',out_)]
    edge_rnn=[('x0','h0'),('x0','h1'),('x0','h2'),
            ('x1','h0'),('x1','h1'),('x1','h2'),
            ('x2','h0'),('x2','h1'),('x2','h2'),
            ('h0','h0'),('h0','h1'),('h0','h2'),
            ('h1','h0'),('h1','h1'),('h1','h2'),
            ('h2','h0'),('h2','h1'),('h2','h2'),
            ('h0','y0'),('h1','y1'),('h2','y2')]
    np_rnn=NetworkProgram(node_rnn,edge_rnn)

    # 重みを指定
    np_rnn.G.add_weighted_edges_from(
        [('x0','h0',1),('x0','h1',10),('x0','h2',0),
        ('x1','h0',0),('x1','h1',1),('x1','h2',0),
        ('x2','h0',0),('x2','h1',0),('x2','h2',1),
        ('h0','h0',0),('h0','h1',-100),('h0','h2',0),
        ('h1','h0',0),('h1','h1',1),('h1','h2',0),
        ('h2','h0',0),('h2','h1',-100),('h2','h2',0),
        ('h0','y0',1),('h1','y1',1),('h2','y2',1)])
    np_rnn.forward([[0,1,0],[1,0,0]])
    """

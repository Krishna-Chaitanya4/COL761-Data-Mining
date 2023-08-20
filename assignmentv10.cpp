#include<bits/stdc++.h>
using namespace std;
class FullNode{
    public:
    int val;
    int count = 0;
    int isLeaf = 0;
    bool exist = false;
    int key;
    map<int,FullNode*> childs;
    FullNode(int n=-1){
        val = n;
    }
    ~FullNode(){
        map<int,FullNode*>:: iterator it = childs.begin();
        while(it!=childs.end()){
            delete it->second;
            it++;
        }
        childs.clear();
    }
};
class FullFPTree{
    public:
    FullNode* root;
    unordered_map<int,int> itemfreq;
    FullFPTree(){
        root = new FullNode();
    }
    ~FullFPTree(){
        delete root;
        itemfreq.clear();
    }
    void Insert(vector<int>& transaction){
        FullNode* curr = root;
        curr->count++;
        int len = transaction.size();
        for(int b=0;b<len;b++){
             int i=transaction[b];
            if(!curr->childs.count(i))curr->childs[i] = new FullNode(i);
            curr->childs[i]->count++;
            curr = curr->childs[i];
        }
        curr->isLeaf++;
    }
    void dfs(FullNode* i,vector<int>& temp,vector<vector<int>>& fina){
        temp.push_back(i->val);
        map<int,FullNode*>:: iterator it = i->childs.begin();
        while(it!=i->childs.end()){
            dfs(it->second,temp,fina);
            it++;
        }
        if(!(int)i->childs.size())fina.push_back(temp);
        temp.pop_back();
    }
    void Print(){
        vector<int> temp;
        vector<vector<int>> fina;
        dfs(root,temp,fina);
        int n = fina.size();
        for(int i=0;i<n;i++){
            int m = fina[i].size();
            for(int j=0;j<m;j++)cout<<fina[i][j]<<" ";
            cout<<endl;
        }
    }
};
class Node{
    public:
    int val;
    int count = 0;
    Node* parent = NULL;
    Node* next = NULL;
    map<int,Node*> childs;
    Node(int n=-1){
        val = n;
    }
    ~Node(){
        map<int,Node*>::iterator it = childs.begin();
        while(it!=childs.end()){
            delete it->second;
            it++;
        }
        childs.clear();
    }
};
class FPTree{
    public:
    Node* root;
    unordered_map<int,int> itemfreq;
    map<pair<int,int>, Node*> headerFile;
    FPTree(){
        root = new Node();
    }
    ~FPTree(){
        delete root;
        itemfreq.clear();
        headerFile.clear();
    }
    void ConditionalInsert(vector<int>& transaction,int threshold,int c){
        Node* curr = root;
        curr->count+=c;
        int len = transaction.size();
        for(int b=0;b<len;b++){
            int i = transaction[b];
            if(itemfreq[i]<threshold || curr->val==i)continue;
            if(!curr->childs.count(i)){
                Node* newnode = new Node(i);
                curr->childs[i] = newnode;
                if(!headerFile.count(make_pair(itemfreq[i],i)))headerFile[make_pair(itemfreq[i],i)]=newnode;
                else{
                    Node* headercurr = headerFile[make_pair(itemfreq[i],i)];
                    while(headercurr->next)headercurr = headercurr->next;
                    headercurr->next = newnode;
                }
            }
            curr->childs[i]->count+=c;
            curr->childs[i]->parent = curr;
            curr = curr->childs[i];
        }
    }
    void dfs(Node* i,vector<int>& temp,vector<vector<int>>& fina){
        temp.push_back(i->val);
        map<int,Node*>::iterator it = i->childs.begin();
        while(it!=i->childs.end()){
            dfs(it->second,temp,fina);
            it++;
        }
        if(!(int)i->childs.size())fina.push_back(temp);
        temp.pop_back();
    }
    void Print(){
        vector<int> temp;
        vector<vector<int>> fina;
        dfs(root,temp,fina);
        int n = fina.size();
        for(int i=0;i<n;i++){
            int m = fina[i].size();
            for(int j=0;j<m;j++)cout<<fina[i][j]<<" ";
            cout<<endl;
        }
    }
};
class CompressorDecompressor{
    public:
    FullFPTree* Tree;
    FPTree* ConTree;
    string fileLocation;
    vector<vector<int>> FrequentItemSet;
    unordered_map<long long,long long> InsertedKeys;
    string outputlocation;
    long long presentkey = -2;
    queue<vector<int>> outputs;

    long long Generatename(long long i){
        if(InsertedKeys.count(i))return InsertedKeys[i];
        InsertedKeys[i] = presentkey;
        presentkey--;
        return InsertedKeys[i];
    }

    CompressorDecompressor(string input,string output){  
        fileLocation = input;
        outputlocation = output;
        Tree = new FullFPTree();
        ConTree = new FPTree();
        if (remove(output.c_str()) != 0);
    }
    void preload(){
        ifstream transactions(fileLocation);
        if(!transactions.is_open())cerr<<"File is not opened"<<endl;
        string s;
        while(getline(transactions,s)){
            stringstream obj(s);
            int var;
            unordered_set<int> se;
            while(obj>>var){
                Tree->itemfreq[var]++;
                if(se.count(var))continue;
                ConTree->itemfreq[var]++;
                se.insert(var);
            }
        }
        transactions.close();
        ifstream transaction(fileLocation);
        if(!transaction.is_open())cerr<<"File is not opened second time."<<endl;
        while(getline(transaction,s)){
            vector<int> temp;
            stringstream obj(s);
            int var;
            while(obj>>var)temp.push_back(var);
            sort(temp.begin(),temp.end(),[this](int& a, int& b){return Tree->itemfreq[a]!=Tree->itemfreq[b]?Tree->itemfreq[a]>Tree->itemfreq[b]:a>b;});
            Tree->Insert(temp);
            sort(temp.begin(),temp.end(),[this](int& a, int& b){return ConTree->itemfreq[a]!=ConTree->itemfreq[b]?ConTree->itemfreq[a]>ConTree->itemfreq[b]:a>b;});
            ConTree->ConditionalInsert(temp,1,1);
        }
        transaction.close();
    }
    FPTree* BuildConditionalTree( Node* end,int threshold){//checked working fine but getting some difference with rutvik values.
        FPTree* ConditionalTree = new FPTree();
        Node* endcurr = end;
        while(endcurr){
            Node* curr = endcurr;
            curr = curr->parent;
            while(curr->val!=-1){
                ConditionalTree->itemfreq[curr->val]+=endcurr->count;
                curr = curr->parent;
            }
            endcurr = endcurr->next;
        }
        endcurr = end;
        while(endcurr){
            Node* curr = endcurr;
            curr = curr->parent;
            vector<int> transaction;
            while(curr->val!=-1){
                transaction.push_back(curr->val);
                curr = curr->parent;
            }
            sort(transaction.begin(),transaction.end(),[&ConditionalTree](const int& a,const int& b){return ConditionalTree->itemfreq[a]!=ConditionalTree->itemfreq[b]? ConditionalTree->itemfreq[a]>ConditionalTree->itemfreq[b]:a>b;});
            ConditionalTree->ConditionalInsert(transaction,threshold,endcurr->count);
            endcurr = endcurr->next;
        }
        return ConditionalTree;
    }
    void Mine(FPTree*& contree,int threshold,vector<int>& prefix){//Checked Working Fine.
        map<pair<int,int>,Node*>::iterator it = contree->headerFile.begin();
        while(it!=contree->headerFile.end()){
            if(contree->itemfreq[it->second->val]<threshold){
                it++;
                continue;
            }
            Node* curr = it->second;
            FPTree* ConditionalTree = BuildConditionalTree(curr,threshold);
            prefix.push_back(curr->val);
            Mine(ConditionalTree,threshold,prefix);
            delete ConditionalTree;
            FrequentItemSet.push_back(prefix);
            prefix.pop_back();
            it++;
        }
    }
    void FrequentItemSetInsertdfs(FullNode* curr,int i,int j){
        if(curr->val!=-1){
            int n = FrequentItemSet[i].size();
            if(j>=n || Tree->itemfreq[FrequentItemSet[i][j]]>Tree->itemfreq[curr->val])return;
            if(FrequentItemSet[i][j]==curr->val)j++;
            if(j==n){
                if(!curr->exist || (curr->exist && n>(int)FrequentItemSet[curr->key].size())){
                    curr->exist = true;
                    curr->key = i;
                }
                return;
            }
        }
        map<int,FullNode*>::iterator it = curr->childs.begin();
        while(it!=curr->childs.end()){
            FrequentItemSetInsertdfs(it->second,i,j);
            it++;
        }
    }
    void FrequentItemSetInsert(){
        int n = FrequentItemSet.size();
        FullNode* curr = Tree->root;
        for(int i=0;i<n;i++){
            sort(FrequentItemSet[i].begin(),FrequentItemSet[i].end(),[this](const int& a,const int& b){return Tree->itemfreq[a]!=Tree->itemfreq[b]? Tree->itemfreq[a]>Tree->itemfreq[b]:a>b;});
            FrequentItemSetInsertdfs(curr,i,0);
        }
    }
    vector<int> CompressedTransaction(vector<int>& transaction,vector<long long> keyslist){
        vector<int> ans;
        sort(keyslist.begin(),keyslist.end(),[this](const int& a,const int& b){return (int)FrequentItemSet[a].size()!=(int)FrequentItemSet[b].size()? (int)FrequentItemSet[a].size()>(int)FrequentItemSet[b].size():a>b;});
        unordered_map<int,int> insertedelements;
        int n = keyslist.size();
        for(int b=0;b<n;b++){
            long long i = keyslist[b];
            bool disjoint = true;
            int len = FrequentItemSet[i].size();
            for(int j=0;j<len;j++)if(insertedelements.count(FrequentItemSet[i][j])){
                disjoint = false;
                break;
            }
            if(!disjoint)continue;
            for(int j=0;j<len;j++)insertedelements[FrequentItemSet[i][j]]++;
            long long name = Generatename(i);
            ans.push_back(name);
        }
        int len = transaction.size();
        for(int b=0;b<len;b++){
            int i = transaction[b];
            if(insertedelements.count(i) && insertedelements[i]>0){
                insertedelements[i]--;
                continue;
            }
            ans.push_back(i);
        }
        return ans;
    }
    void transactiondfs(FullNode* curr,vector<int>& transaction,vector<long long>& keyslist){
        if(curr->isLeaf){
            vector<int> s = CompressedTransaction(transaction,keyslist);
            int k = curr->isLeaf;
            while(k--)outputs.push(s);
        }
        if(curr->val!=-1)transaction.push_back(curr->val);
        if(curr->exist)keyslist.push_back(curr->key);
        map<int,FullNode*>::iterator it = curr->childs.begin();
        while(it!= curr->childs.end()){
            transactiondfs(it->second,transaction,keyslist);
            it++;
        }
        if(curr->val!=-1)transaction.pop_back();
        if(curr->exist)keyslist.pop_back();
    }
    void write(){
        ofstream ofile(outputlocation,ios::app);
        if(!ofile.is_open())cerr<<"File is not opened"<<endl;
        int k = outputs.size();
        while(k--){
            vector<int> a = outputs.front();
            outputs.pop();
            int len = a.size();
            for(int j=0;j<len;j++)ofile<<a[j]<<" ";
            ofile<<endl;
        }
        ofile<<-1<<endl;
        unordered_map<long long,long long>::iterator it = InsertedKeys.begin();
        while(it!= InsertedKeys.end()){
            ofile<<it->second<<" ";
            int len = FrequentItemSet[it->first].size();
            for(int j=0;j<len;j++)ofile<<FrequentItemSet[it->first][j]<<" ";
            ofile<<endl;
            it++;
        }
        ofile.close();
    }
    int generatethreshold(){
        int numerator = 0;
        int denominator = 0;
        unordered_map<int,int>::iterator it = Tree->itemfreq.begin();
        while(it!=Tree->itemfreq.end()){
            denominator++;
            numerator+=it->second;
            it++;
        }
        int ans = numerator/denominator;
        return ans;
    }
    void Compressor(int threshold){
        vector<int> prefix;
        Mine(ConTree,threshold,prefix);
        delete ConTree;
        FrequentItemSetInsert();
        vector<int> transaction;
        vector<long long> keyslist;
        transactiondfs(Tree->root,transaction,keyslist);
        write();
    }
};

int main(string input,string output){
    CompressorDecompressor com(input,output);
    com.preload();
    int threshold = com.generatethreshold();
    com.Compressor(threshold);
    return 0;
}

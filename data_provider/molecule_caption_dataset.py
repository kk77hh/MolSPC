import torch
from torch_geometric.data import Dataset
import os
import pandas as pd
import numpy as np
import re


def extract_target_properties(text):
    
    # 修改正则表达式以支持负数
    pattern = r"\[(\w+): (-?[\d.]+)\]"
    matches = re.findall(pattern, text)

    # 提取目标分子的属性值（第二个值）
    target_properties = [f"[{prop}: {target}]" for prop, target in matches]

    # 拼接为目标格式字符串
    return "; ".join(target_properties)

def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)

        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        return len(subdirectories)
    except FileNotFoundError:
        print(f"文件夹 '{folder_path}' 不存在。")
        return -1  # 返回 -1 表示文件夹不存在
    except Exception as e:
        print(f"发生错误：{e}")
        return -2  # 返回 -2 表示发生了其他错误
class MoleculeCaption(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list) + '\n'

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return data_graph, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token




class MoleculeCaption_double_pretrain(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_pretrain, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        try:
            graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
            smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
            graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
            smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
            text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        except:
            index = index+1
            try:
                graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
                smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
                graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
                smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
                text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
            except:
                index = index+1
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        data_graph_0 = torch.load("graph_data.pt")
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        #smiles_prompt = smiles_prompt1+"The front is the source molecule, followed by the target molecule."+smiles_prompt2+"What are the property values of these two drugs?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        
        text_list = []
        count = 0
        smiles_prompt_compare = smiles_prompt1+". The front is the source molecule, followed by the target molecule."+smiles_prompt2+". What are the property and structure of these two drugs?"
        #smiles_prompt_structure = smiles_prompt1+". The front is the source molecule, followed by the target molecule."+smiles_prompt2+". What are the structure differences of these two drugs?"

        for line in open(text_path, 'r', encoding='utf-8'):
            
            count += 1
            text_list.append(line.strip('\n'))
            
            if count > 100:
                break
        text_compare = ''.join(text_list[0])
        '''text_structure = ''.join(text_list[1])
        text_structure = text_structure+(text_list[2])
        text_compare = text_compare+text_structure'''
        text_opt = ''.join(text_list[1])
        
        opt_property=extract_target_properties(text_list[0])
        smiles_prompt_opt = "Please generate a molecule according to the following property description: %s ."%(opt_property)
        smiles_prompt_opt = smiles_prompt_opt+"[START_I_SMILES]C[END_I_SMILES]"
        smiles_prompt_opt = smiles_prompt_opt+"[START_I_SMILES]C[END_I_SMILES]"
        print("--------------------------------------")
        print(smiles_prompt_compare)
        print(text_compare)
        '''print(smiles_prompt_structure)
        print(text_structure)'''
        print(smiles_prompt_opt)
        print(text_opt)
        return data_graph1,data_graph2,data_graph_0, text_compare, text_opt , smiles_prompt_compare,smiles_prompt_opt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token





class MoleculeCaption_double_finetune(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_finetune, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        value_name_list = os.listdir(self.root+'value/'+str(index)+'/')

        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        #smiles_prompt = smiles_prompt1+"The front is the source molecule, followed by the target molecule."+smiles_prompt2+"What are the property values of these two drugs?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        value_path = os.path.join(self.root, 'value/'+str(index)+'/', value_name_list[0])

        with open(value_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            
            opt_property=extract_target_properties(line.strip('\n'))

        
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            #smiles_prompt1 = smiles_prompt1+','
            
            smiles_prompt=smiles_prompt1+". "+"The front is the source molecule. Please change the property of %s to the value as follows: %s."%(smiles_prompt1,opt_property)
            if count > 100:
                break
        text = ' '.join(text_list)
        
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
   
   
        return sentence_token




from torch.utils.data import Dataset
from torch_geometric.loader.dataloader import Collater
from rdkit import Chem

class InferenceDataset(Dataset):
    def __init__(self, smiles, prompt, graph_func):
        self.smiles = smiles
        self.prompt = prompt
        self.graph_func = graph_func

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        graph = self.graph_func(self.smiles)

        return graph,graph, "", self.prompt




class MoleculeCaption_double_value(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_value, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
            #return 100
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+" what is the solvation Gibbs free energy of this pair of molecules?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token    
    

class MoleculeCaption_universal(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_universal, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        #query_name_list = os.listdir(self.root+'query/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt

        '''query_path = os.path.join(self.root, 'query/'+str(index)+'/', query_name_list[0])
        with open(query_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            query = lines[0].strip()
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> . '+query'''
        smiles_prompt = smiles_prompt1+"The front is the first molecule, followed by the second molecule."+smiles_prompt2+"What are the side effects of these two drugs?"
        # load and process text

        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token  
    
if __name__ == '__main__':
    import numpy as np
    pretrain = MoleculeCaption('../data/PubChemDataset_v4/pretrain/', 1000, '')
    train = MoleculeCaption('../data/PubChemDataset_v4/train/', 1000, '')
    valid = MoleculeCaption('../data/PubChemDataset_v4/valid/', 1000, '')
    test = MoleculeCaption('../data/PubChemDataset_v4/test/', 1000, '')

    for subset in [pretrain, train, valid, test]:
        g_lens = []
        t_lens = []
        for i in range(len(subset)):  
            data_graph, text, _ = subset[i]
            g_lens.append(len(data_graph.x))
            t_lens.append(len(text.split()))
            # print(len(data_graph.x))
        g_lens = np.asarray(g_lens)
        t_lens = np.asarray(t_lens)
        print('------------------------')
        print(g_lens.mean())
        print(g_lens.min())
        print(g_lens.max())
        print(t_lens.mean())
        print(t_lens.min())
        print(t_lens.max())

import os
import re
import torch
import random
import pickle
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors  # 添加导入
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain
import signal
import time
import torch.nn.functional as F
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles
from rdkit.Chem import MolToInchi
from shutil import copy
from rdkit.Chem.FilterCatalog import GetFunctionalGroupHierarchy
from rdkit.Chem import RDConfig
import multiprocessing
from rdkit.Chem import rdFMCS



class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()


def find_mcs_smiles(smiles1, smiles2):
    # 将 SMILES 转换为 RDKit 分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("输入的 SMILES 字符串无效，无法解析为分子对象。")

    # 设置 MCS 参数
    params = rdFMCS.MCSParameters()
    params.UseBondTypes = True
    params.UseChirality = True
    params.MatchValence = True

    # 计算 MCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2], params)

    # 获取 MCS 的 SMARTS 字符串
    mcs_smarts = mcs_result.smartsString

    # 将 SMARTS 转换为 SMILES
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    if mcs_mol is None:
        return None  # 如果无法生成分子对象，返回 None

    return Chem.MolToSmiles(mcs_mol)

# 设置超时信号处理器
signal.signal(signal.SIGALRM, timeout_handler)


def find_mcs_smiles_with_timeout(smiles1, smiles2, timeout=1):
    def worker(smiles1, smiles2, output):
        try:
            result = find_mcs_smiles(smiles1, smiles2)
            output.put(result)
        except Exception as e:
            output.put(None)

    output = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(smiles1, smiles2, output))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"计算 MCS 超时，跳过分子对: {smiles1}, {smiles2}")
        return None

    return output.get()


def smi_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smi = Chem.MolToSmiles(mol) # standardize
        mol = Chem.MolFromSmiles(smi)
        return mol
    except:
        return None

def get_functional_groups(mol):
    functionalGroups = GetFunctionalGroupHierarchy()
    fgs = [
        match.filterMatch.GetName() for match in functionalGroups.GetFilterMatches(mol)
    ]
    return fgs




def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    # the fragment genereated from smarts would have a redundant carbon, here to remove the redundant carbon
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash


def return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    mol = Chem.MolFromSmiles(smiles)
    hit_at = []
    hit_fg_name = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash
    # sort function group atom by atom number
    sorted_all_hit_fg_at = sorted(all_hit_fg_at,
                                  key=lambda fg: len(fg),
                                  reverse=True)
    # remove small function group (wrongly matched), they are part of other big function groups
    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)
    # wash the hit function group atom by using the remained fg, remove the small wrongly matched fg
    hit_at_wash = []
    hit_fg_name_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_name_wash.append(hit_fg_name[j])
    return hit_at_wash, hit_fg_name_wash




def extract_functional_groups(smiles):
    """
    输入分子 SMILES 字符串，输出分子官能团信息。

    参数:
        smiles (str): 分子的 SMILES 字符串。

    返回:
        hit_fg_name_wash (list): 匹配到的官能团名称列表。
    """
    # 加载官能团配置
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_without_ca_smart = [
        '[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
        'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
        '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
        '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
        '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
        '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
        '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
        '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
        '[C;D4]([C;D1])([C;D1])-[C;D1]', '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1',
        '[O;D2]-[C;D2]-[C;D1;H3]', '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]'
    ]
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]
    fg_name_list = [
        '*NC(=O)C', '*C(=O)O', '*C(=O)OC', '*C(=O)', '*C(=O)N', '*C(=O)C', '*N=C=O', '*N=C=S', '*N(O)O', '*N=O', '*=NO', '*=NC',
        '*N=C', '*N=NC', '*N=N', '*N#N', '*C#N', '*S(=O)(=O)N', '*NS(=O)(=O)C', '*S(=O)(=O)O', '*S(=O)(=O)OC', '*S(=O)(=O)C', '*S(=O)(=O)Cl',
        '*S(=O)C', '*SC', '*S', '*=S', '*[X]', '*C(C)(C)C', '*C(F)(F)F', '*C#C', '*C1CC1', '*OCC', '*OC', '*O', '*=O', '*N', '*=N', '*#N'
    ]

    # 调用已有函数提取官能团信息
    hit_at_wash, hit_fg_name_wash = return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)

    return hit_fg_name_wash




def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges

def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(smiles):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def extract_texts_from_csv(csv_path, id1, id2):
    text1, text2 = None, None

    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',', 1)  # Split only at the first comma
            if len(parts) == 2:
                current_id, text = parts
                if current_id == id1:
                    text1 = text
                elif current_id == id2:
                    text2 = text
            if text1 is not None and text2 is not None:
                break  # Stop searching if both texts are found

    return text1, text2

def calculate_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    properties = {
        "MW": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "LogP": Descriptors.MolLogP(mol),
        "QED": Descriptors.qed(mol)
    }
    return properties


data = pd.read_csv("datasets/dataset_train.csv")
#data = data.sample(n=10000, random_state=42).reset_index(drop=True)  # 使用 pandas 的 sample 方法
data = np.array(data)
idx = np.random.permutation(len(data))
train_idx=idx


for i in range(len(train_idx)):

    print("Processing train pair {}/{}".format(i+1, len(train_idx)))

    data[train_idx[i]]
    smiles1 = data[train_idx[i]][0]
    smiles2 = data[train_idx[i]][1]
    mol = Chem.MolFromSmiles(smiles1)

    # 使用带超时的 MCS 计算函数
    mcs = find_mcs_smiles_with_timeout(smiles1, smiles2)
    

    # 如果 MCS 计算超时或失败，则跳过该分子对
    if mcs is None:
        scaffold = Chem.MurckoDecompose(mol)
        mcs = Chem.MolToSmiles(scaffold)
    print(mcs)
    try:
        mcs = Chem.MolToSmiles(Chem.MolFromSmiles(mcs))
    except:
        scaffold = Chem.MurckoDecompose(mol)
        mcs = Chem.MolToSmiles(scaffold)

    os.makedirs("data/compare/train/smiles1/"+str(i))
    os.makedirs("data/compare/train/smiles2/"+str(i))
    os.makedirs("data/compare/train/graph1/"+str(i))
    os.makedirs("data/compare/train/graph2/"+str(i))
    os.makedirs("data/compare/train/text/"+str(i))


    data1 = mol_to_graph_data_obj_simple(data[train_idx[i]][0])
    torch.save(data1,"data/compare/train/graph1/"+str(i)+'/graph_data.pt')
    data1 = mol_to_graph_data_obj_simple(data[train_idx[i]][1])
    torch.save(data1,"data/compare/train/graph2/"+str(i)+'/graph_data.pt')
    
    smiles1 = data[train_idx[i]][0]
    smiles2 = data[train_idx[i]][1]




    fg_smi1 = get_functional_groups(smi_to_mol(smiles1))
    fg_smi2 = get_functional_groups(smi_to_mol(smiles2))
    file = open("data/compare/train/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/compare/train/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()

    

    # Calculate molecular properties for smiles1 and smiles2
    props1 = calculate_molecular_properties(smiles1)
    props2 = calculate_molecular_properties(smiles2)

    
    pro_change = ''
    # Add new molecular properties to pro_change
    if props1 and props2:
        pro_change += '[MW: {:.2f}, {:.2f}]; '.format(props1["MW"], props2["MW"])
        pro_change += '[TPSA: {:.2f}, {:.2f}]; '.format(props1["TPSA"], props2["TPSA"])
        pro_change += '[HBD: {}, {}]; '.format(props1["HBD"], props2["HBD"])
        pro_change += '[HBA: {}, {}]; '.format(props1["HBA"], props2["HBA"])
        pro_change += '[RB: {}, {}]; '.format(props1["RB"], props2["RB"])
        pro_change += '[LogP: {:.2f}, {:.2f}]; '.format(props1["LogP"], props2["LogP"])
        pro_change += '[QED: {:.2f}, {:.2f}]; '.format(props1["QED"], props2["QED"])
        #pro_change += '[SA: {:.2f}, {:.2f}];'.format(props1["SA"], props2["SA"])  # 新增

    #text = "The %s property value of the source molecule is %s and the %s property value of the target molecule is %s."%(task_name,"{:.3f}".format(float(source_prop)),task_name,"{:.3f}".format(float(target_prop)))
    text = "The property values of the source molecule and target molecule are: %s . "%(pro_change)
    text4 = "%s"%('[START_I_SMILES]{}[END_I_SMILES].'.format(smiles2))

    text3 = text
    text4 = text4+'\n'


    text_mcs = 'Maximum Common Substructure: %s . ' % (mcs)
    text_fg = 'The functional group of the source molecule is %s and the functional group of the target molecule is %s.'%(fg_smi1,fg_smi2)
    text3 =text3+ text_mcs
    text3 =text3+ text_fg + '\n'

    file = open("data/compare/train/text/"+str(i)+"/text.txt","w")
    file.write(text3)
    
    file.write(text4)

    file.close()
    






data = pd.read_csv("datasets/dataset_test.csv")
data = data.sample(n=2000, random_state=42).reset_index(drop=True)  # 使用 pandas 的 sample 方法
data = np.array(data)
idx = np.random.permutation(len(data))
valid_idx=idx

for i in range(len(valid_idx)):
    
    print("Processing valid pair {}/{}".format(i+1, len(valid_idx)))

    data[valid_idx[i]]

    smiles1 = data[valid_idx[i]][0]
    smiles2 = data[valid_idx[i]][1]

    mol = Chem.MolFromSmiles(smiles1)



    # 使用带超时的 MCS 计算函数
    mcs = find_mcs_smiles_with_timeout(smiles1, smiles2)
    

    # 如果 MCS 计算超时或失败，则跳过该分子对
    if mcs is None:
        scaffold = Chem.MurckoDecompose(mol)
        mcs = Chem.MolToSmiles(scaffold)
    print(mcs)
    try:
        mcs = Chem.MolToSmiles(Chem.MolFromSmiles(mcs))
    except:
        scaffold = Chem.MurckoDecompose(mol)
        mcs = Chem.MolToSmiles(scaffold)

    os.makedirs("data/compare/valid/smiles1/"+str(i))
    os.makedirs("data/compare/valid/smiles2/"+str(i))
    os.makedirs("data/compare/valid/graph1/"+str(i))
    os.makedirs("data/compare/valid/graph2/"+str(i))
    os.makedirs("data/compare/valid/text/"+str(i))


    data1 = mol_to_graph_data_obj_simple(data[valid_idx[i]][0])
    torch.save(data1,"data/compare/valid/graph1/"+str(i)+'/graph_data.pt')
    data1 = mol_to_graph_data_obj_simple(data[valid_idx[i]][1])
    torch.save(data1,"data/compare/valid/graph2/"+str(i)+'/graph_data.pt')


    fg_smi1 = get_functional_groups(smi_to_mol(smiles1))
    fg_smi2 = get_functional_groups(smi_to_mol(smiles2))
    
    
    file = open("data/compare/valid/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/compare/valid/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
    
    '''task_name = data[valid_idx[i]][6]
    task_name = task_name.split('+')'''

    # Calculate molecular properties for smiles1 and smiles2
    props1 = calculate_molecular_properties(smiles1)
    props2 = calculate_molecular_properties(smiles2)

    
    pro_change = ''
    # Add new molecular properties to pro_change
    if props1 and props2:
        pro_change += '[MW: {:.2f}, {:.2f}]; '.format(props1["MW"], props2["MW"])
        pro_change += '[TPSA: {:.2f}, {:.2f}]; '.format(props1["TPSA"], props2["TPSA"])
        pro_change += '[HBD: {}, {}]; '.format(props1["HBD"], props2["HBD"])
        pro_change += '[HBA: {}, {}]; '.format(props1["HBA"], props2["HBA"])
        pro_change += '[RB: {}, {}]; '.format(props1["RB"], props2["RB"])
        pro_change += '[LogP: {:.2f}, {:.2f}]; '.format(props1["LogP"], props2["LogP"])
        pro_change += '[QED: {:.2f}, {:.2f}]; '.format(props1["QED"], props2["QED"])
        #pro_change += '[SA: {:.2f}, {:.2f}];'.format(props1["SA"], props2["SA"])  # 新增
    
    #text = "The %s property value of the source molecule is %s and the %s property value of the target molecule is %s."%(task_name,"{:.3f}".format(float(source_prop)),task_name,"{:.3f}".format(float(target_prop)))
    text = "The property values of the source molecule and target molecule are: %s . "%(pro_change)
    text4 = "%s"%('[START_I_SMILES]{}[END_I_SMILES].'.format(smiles2))

    text3 = text
    text4 = text4+'\n'
    text_mcs = 'Maximum Common Substructure: %s . ' % (mcs)
    text_fg = 'The functional group of the source molecule is %s and the functional group of the target molecule is %s.'%(fg_smi1,fg_smi2)
    text3 = text3+text_mcs
    text3 = text3+text_fg + '\n'
    file = open("data/compare/valid/text/"+str(i)+"/text.txt","w")
    file.write(text3)
    '''file.write(text_mcs)
    file.write(text_fg)'''
    file.write(text4)
    file.close()
    

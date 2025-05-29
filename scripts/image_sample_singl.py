"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th

import io
import PIL.Image as Image
import drawSvg as drawsvg
import cairosvg
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion import dist_util, logger
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
import webcolors
import networkx as nx
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos

# import random
# th.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

import torch
import numpy as np
from termcolor import colored


room_labels = { 
    1: "LivingRoom", 
    2: "Kitchen", 
    3: "Bedroom", 
    4: "Bathroom", 
    5: "Balcony", 
    6: "Entrance", 
    7: "DiningRoom", 
    8: "StudyRoom", 
    10: "Storage", 
     
    13: "External", 
    15: "FrontDoor", 
    16: "Unknown", 
    17: "InteriorDoor", 
}


def print_model_inputs(model_kwargs, idx, json_filename):
    """
    Выводит информацию о синтетических входных данных модели для генерации в одиночном режиме.
    Показывает, что координаты отсутствуют (нулевые), и выводит граф связей (10 рёбер), маски и другие условия из rplan_eval_8_syn.npz.
    """
    print("\n" + "="*80)
    print(colored(f"Синтетические входные данные модели для генерации плана #{idx} (JSON: {json_filename})", "cyan", attrs=["bold"]))
    print("="*80)

    # 1. Координаты углов
    print(colored("\n1. Координаты углов:", "yellow", attrs=["bold"]))
    print(colored("  - Координаты отсутствуют (все нулевые, модель начинает с топологии)!", "green"))

    # 2. Граф связей (syn_graph)
    print(colored("\n2. Граф связей (syn_graph):", "yellow", attrs=["bold"]))
    syn_graph = model_kwargs.get('syn_graph', None)
    if syn_graph is not None and isinstance(syn_graph, torch.Tensor):
        print(f"  - Форма: {list(syn_graph.shape)}")
        print(f"  - Тип данных: {syn_graph.dtype}")
        non_zero_edges = syn_graph[syn_graph.sum(dim=-1) != 0].cpu().numpy()
        print(f"  - Количество рёбер в графе: {len(non_zero_edges)}")
        print(f"  - Пример рёбер (первые 10):")
        for edge in non_zero_edges[:10]:
            print(f"      {edge}")
    else:
        print(colored("  - Граф связей (syn_graph) отсутствует!", "red"))

    # 3. Маски внимания (syn_door_mask, syn_self_mask, syn_gen_mask)
    print(colored("\n3. Маски внимания:", "yellow", attrs=["bold"]))
    for mask_name in ['syn_door_mask', 'syn_self_mask', 'syn_gen_mask']:
        mask = model_kwargs.get(mask_name, None)
        print(f"    - {mask_name}:")
        if mask is not None and isinstance(mask, torch.Tensor):
            print(f"        Форма: {list(mask.shape)}")
            print(f"        Тип данных: {mask.dtype}")
            non_zero = (mask != 0).sum().item()
            print(f"        Количество ненулевых элементов: {non_zero}")
        else:
            print(colored(f"        Отсутствует!", "red"))

    # 4. Дополнительные условия (syn_room_types, syn_connections, syn_corner_indices, syn_room_indices, syn_src_key_padding_mask)
    print(colored("\n4. Дополнительные условия:", "yellow", attrs=["bold"]))
    for key in ['syn_room_types', 'syn_connections', 'syn_corner_indices', 'syn_room_indices', 'syn_src_key_padding_mask']:
        value = model_kwargs.get(key, None)
        print(f"    - {key}:")
        if value is not None and isinstance(value, torch.Tensor):
            print(f"        Форма: {list(value.shape)}")
            print(f"        Тип данных: {value.dtype}")
            if key == 'syn_room_types':
                room_types = torch.argmax(value[0], dim=-1).cpu().numpy()
                non_zero_types = room_types[room_types != 0]
                print(f"        Типы комнат (первые 5 ненулевых): {non_zero_types[:5]}")
            elif key == 'syn_connections':
                connections = value[0].cpu().numpy()
                print(f"        Пример связей углов (первые 2): {connections[:2]}")
            elif key == 'syn_corner_indices':
                corner_indices = torch.argmax(value[0], dim=-1).cpu().numpy()
                non_zero_indices = corner_indices[corner_indices != 0]
                print(f"        Индексы углов (первые 5 ненулевых): {non_zero_indices[:5]}")
            elif key == 'syn_room_indices':
                room_indices = torch.argmax(value[0], dim=-1).cpu().numpy()
                non_zero_indices = room_indices[room_indices != 0]
                print(f"        Индексы комнат (первые 5 ненулевых): {non_zero_indices[:5]}")
            elif key == 'syn_src_key_padding_mask':
                non_zero = (value[0] != 0).sum().item()
                print(f"        Количество ненулевых элементов: {non_zero}")
        else:
            print(colored(f"        Отсутствует!", "red"))

    print("\n" + "="*80)


def find_plan_index_in_npz(npz_path, json_filename):
    """
    Находит индекс записи (квартиры) по имени json-файла внутри npz.
    Возвращает индекс, либо кидает ошибку, если не найдено.
    """
    data = np.load(npz_path, allow_pickle=True)
    filenames = data['filenames']
    filenames = [str(f) for f in filenames]
    target_basename = os.path.basename(json_filename)
    for idx, f in enumerate(filenames):
        if os.path.basename(f) == target_basename:
            return idx
    raise ValueError(f"Файл {json_filename} не найден в {npz_path}.")

def get_plan_from_npz(npz_path, idx):
    """
    Возвращает все массивы (ключ-значение) по нужному индексу из npz.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key][idx] for key in data.files}


bin_to_int = lambda x: int("".join([str(int(i.cpu().data)) for i in x]), 2)
def bin_to_int_sample(sample, resolution=256):
    sample_new = th.zeros([sample.shape[0], sample.shape[1], sample.shape[2], 2])
    sample[sample<0] = 0
    sample[sample>0] = 1
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                sample_new[i, j, k, 0] = bin_to_int(sample[i, j, k, :8])
                sample_new[i, j, k, 1] = bin_to_int(sample[i, j, k, 8:])
    sample = sample_new
    sample = sample/(resolution/2) - 1
    return sample

def get_graph(indx, g_true, ID_COLOR, draw_graph, save_svg):
    # build true graph
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label
        if _type >= 0 and _type not in [11, 12]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    # add edges
    for k, m, l in g_true[1]:
        k = int(k)
        l = int(l)
        _type_k = g_true[0][k]
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [11, 12] and _type_l not in [11, 12]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        elif m > 0 and (_type_k==11 or _type_l==11):
            if _type_k==11:
                G_true.add_edges_from([(l, -1)])
            else:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    if draw_graph:
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
        nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
                font_weight='bold', edgecolors=edgecolors, width=4.0, with_labels=False)
        if save_svg:
            plt.savefig(f'outputs/graphs_gt/{indx}.svg')
        else:
            plt.savefig(f'outputs/graphs_gt/{indx}.jpg')
        plt.close('all')
    return G_true

def estimate_graph(indx, polys, nodes, G_gt, ID_COLOR, draw_graph, save_svg):
    nodes = np.array(nodes)
    G_gt = G_gt[1-th.where((G_gt == th.tensor([0,0,0], device='cpu')).all(dim=1))[0]]
    G_gt = get_graph(indx, [nodes, G_gt], ID_COLOR, draw_graph, save_svg)
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    edge_labels = {}
    # add nodes
    for k, label in enumerate(nodes):
        _type = label
        if _type >= 0 and _type not in [11, 12]:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            linewidths.append(0.0)
    # add outside node
    G_estimated.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    # add node-to-door connections
    doors_inds = np.where((nodes == 11) | (nodes == 12))[0]
    rooms_inds = np.where((nodes != 11) & (nodes != 12))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:
                p1, p2 = polys[k], polys[l]
                p1, p2 = Polygon(p1), Polygon(p2)
                if not p1.is_valid:
                    p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
                if not p2.is_valid:
                    p2 = geom_factory(lgeos.GEOSMakeValid(p2._geom))
                iou = p1.intersection(p2).area/ p1.union(p2).area
                if iou > 0 and iou < 0.2:
                    doors_rooms_map[k].append((l, iou))
    # draw connections
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k] != 11:
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                edge_labels[(l1, l2)] = k
                G_estimated.add_edges_from([(l1, l2)])
        else:
            if len(_conn) > 0:
                l1 = _conn[0][0]
                edge_labels[(-1, l1)] = k
                G_estimated.add_edges_from([(-1, l1)])
    # add missed edges
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])
    # add edges colors
    colors = []
    mistakes = 0
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')
    if draw_graph:
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
        weights = [4 for u, v in G_estimated_complete.edges()]
        nx.draw(G_estimated_complete, pos, edge_color=colors, linewidths=linewidths, edgecolors=edgecolors, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', width=weights, with_labels=False)
        if save_svg:
            plt.savefig(f'outputs/graphs_pred/{indx}.svg')
        else:
            plt.savefig(f'outputs/graphs_pred/{indx}.jpg')
        plt.close('all')
    return mistakes

def to_scalar(x):
    """Безопасно получить скаляр из тензора/массива или вернуть само значение."""
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, np.ndarray) and x.shape == ():  # np scalar
        return x.item()
    return x

def save_samples(
        sample, ext, model_kwargs, 
        tmp_count, num_room_types, 
        save_gif=True, save_edges=False,
        door_indices=[11, 12, 13], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False):
    prefix = 'syn_' if is_syn else ''
    graph_errors = []
    if not save_gif:
        sample = sample[-1:]
    for i in tqdm(range(sample.shape[1])):
        resolution = 256
        legend_width = 200
        canvas_width = resolution + legend_width
        images, images2, images3 = [], [], []
        for k in range(sample.shape[0]):
            draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw2.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw3.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw_color = drawsvg.Drawing(canvas_width, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))
            draw_color.append(drawsvg.Rectangle(resolution,0,legend_width,resolution, fill='#F9F9F9'))

            polys, types = [], []
            c = 0
            for j, point in enumerate(sample[k][i]):
                mask_tensor = model_kwargs[f'{prefix}src_key_padding_mask']
                if hasattr(mask_tensor, 'ndim'):
                    ndim = mask_tensor.ndim
                elif isinstance(mask_tensor, np.ndarray):
                    ndim = mask_tensor.ndim
                else:
                    ndim = 0
                if ndim == 0:
                    mask_val = to_scalar(mask_tensor)
                elif ndim == 1:
                    mask_val = to_scalar(mask_tensor[j])
                else:
                    mask_val = to_scalar(mask_tensor[i, j])
                if mask_val == 1:
                    continue

                point = point.cpu().data.numpy()
                if j == 0:
                    poly = []
                prev_ri = model_kwargs[f'{prefix}room_indices'][i, j-1] if j > 0 else None
                cur_ri = model_kwargs[f'{prefix}room_indices'][i, j]
                if j > 0 and (cur_ri != prev_ri).any():
                    polys.append(poly)
                    types.append(c)
                    poly = []
                point = (point/2 + 0.5) * resolution
                poly.append((point[0], point[1]))
                if j > 0:
                    c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j-1].cpu().numpy())
                else:
                    c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j].cpu().numpy())
            polys.append(poly)
            types.append(c)

            used_types = set()
            for poly, c in zip(polys, types):
                if c in door_indices or c == 0:
                    continue
                used_types.add(c)
                room_type = c
                c_rgb = webcolors.hex_to_rgb(ID_COLOR[c])
                coords = np.array(poly).flatten().tolist()
                draw_color.append(drawsvg.Lines(*coords, close=True,
                    fill=ID_COLOR[room_type], fill_opacity=1.0,
                    stroke='black', stroke_width=1))
                half_hex = webcolors.rgb_to_hex([int(x/2) for x in c_rgb])
                draw.append(drawsvg.Lines(*coords, close=True, fill='black', fill_opacity=0.0,
                    stroke=half_hex, stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*coords, close=True,
                    fill=ID_COLOR[room_type], fill_opacity=1.0,
                    stroke=half_hex, stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256),
                        fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256),
                        fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                used_types.add(c)
                room_type = c
                coords = np.array(poly).flatten().tolist()
                draw_color.append(drawsvg.Lines(*coords, close=True,
                    fill=ID_COLOR[room_type], fill_opacity=1.0,
                    stroke='black', stroke_width=1))
                half_hex = webcolors.rgb_to_hex([int(x/2) for x in c_rgb])
                draw.append(drawsvg.Lines(*coords, close=True, fill='black', fill_opacity=0.0,
                    stroke=half_hex, stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*coords, close=True,
                    fill=ID_COLOR[room_type], fill_opacity=1.0,
                    stroke=half_hex, stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256),
                        fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256),
                        fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))

            legend_x, legend_y, entry_h, offs = resolution + 20, 250, 24, 35
            draw_color.append(drawsvg.Text('Legend', 16,
                legend_x, legend_y-entry_h - 10,
                text_anchor='start', font_family='sans-serif',
                font_weight='bold', fill='#333'))
            for idx, rid in enumerate(sorted(used_types), start=1):
                y = legend_y - idx * entry_h - offs
                draw_color.append(drawsvg.Rectangle(legend_x, y-5, 16, 16,
                    fill=ID_COLOR.get(rid, '#CCCCCC'),
                    stroke='#666', stroke_width=0.5))
                draw_color.append(drawsvg.Text(room_labels.get(rid, ''),
                    12, legend_x+22, y,
                    text_anchor='start', font_family='sans-serif',
                    fill='#333'))

            images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))))
            images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg()))))
            images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg()))))
            if save_edges:
                draw.saveSvg(f'outputs/{ext}/{tmp_count+i}_{k}_{ext}.svg')
            if save_svg:
                draw_color.saveSvg(f'outputs/{ext}/{tmp_count+i}c_{k}_{ext}.svg')
            else:
                Image.open(io.BytesIO(cairosvg.svg2png(draw_color.asSvg()))).save(
                    f'outputs/{ext}/{tmp_count+i}c_{ext}.png')
            if k == sample.shape[0] - 1:
                if 'graph' in model_kwargs:
                    graph_errors.append(estimate_graph(
                        tmp_count + i, polys, types,
                        model_kwargs[f'{prefix}graph'][i],
                        ID_COLOR=ID_COLOR, draw_graph=draw_graph,
                        save_svg=save_svg))
                else:
                    graph_errors.append(0)
        if save_gif:
            imageio.mimwrite(f'outputs/gif/{tmp_count+i}.gif',
                images, fps=10, loop=1)
            imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v2.gif',
                images2, fps=10, loop=1)
            imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v3.gif',
                images3, fps=10, loop=1)
    return graph_errors



def batchify_single_plan(arr, cond):
    import torch as th
    if isinstance(arr, np.ndarray):
        arr = th.from_numpy(arr).float()
    if arr.ndim == 2:
        arr = arr.unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] != 1:
        arr = arr.unsqueeze(0)
    data_sample = arr

    model_kwargs = {}
    for key, v in cond.items():
        # Чиним object-поля!
        if isinstance(v, np.ndarray) and v.dtype == np.object_:
            # Для масок и индексов (shape N, M)
            if v.shape[0] > 0 and hasattr(v[0], 'shape'):
                try:
                    v = np.stack(v)
                except Exception:
                    v = np.array(v.tolist())
            v = v.astype(np.float32)  # если нужно int, поменяй тут
        # connections — иногда нужны int!
        if 'connections' in key or 'graph' in key:
            v = np.array(v)
            # Обычно (100, 2) или (200, 3)
            # connections нужны int64 (для индексов), graph можно float64 если значения
            if v.dtype != np.float32 and v.dtype != np.float64:
                v = v.astype(np.int64)
            else:
                v = v.astype(np.float32)
        # Остальные float32
        elif isinstance(v, np.ndarray) and v.dtype != np.object_:
            v = v.astype(np.float32)
        # torchify
        if isinstance(v, np.ndarray):
            v = th.from_numpy(v)
        if isinstance(v, th.Tensor) and v.ndim == 2:
            v = v.unsqueeze(0)
        model_kwargs[key] = v
    return data_sample, model_kwargs

def main():

    os.makedirs('outputs/pred', exist_ok=True)
    os.makedirs('outputs/gt', exist_ok=True)
    os.makedirs('outputs/gif', exist_ok=True)
    os.makedirs('outputs/graphs_gt', exist_ok=True)
    os.makedirs('outputs/graphs_pred', exist_ok=True)

    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # model.to(dist_util.dev())
    model.eval()

    if hasattr(args, "single_json_file") and args.single_json_file:
        # --- все, что нужно для single mode, определяем тут ---
        from house_diffusion.rplanhg_datasets import RPlanhgDataset
        ID_COLOR = {
            1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
            6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
            13: '#785A67', 12: '#D3A2C7'
        }
        num_room_types = 14

        dataset = RPlanhgDataset(args.set_name, args.analog_bit, args.target_set)
        npz_path = f"processed_rplan/rplan_{args.set_name}_{args.target_set}.npz"
        idx = find_plan_index_in_npz(npz_path, args.single_json_file)

        arr, cond = dataset[idx]
    
        data_sample, model_kwargs = batchify_single_plan(arr, cond)
        #print_model_inputs(model_kwargs, idx, args.single_json_file)

        
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model,
            data_sample.shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            analog_bit=args.analog_bit,
        )

        sample_gt = data_sample.unsqueeze(0)
        sample = sample.permute([0, 1, 3, 2])
        sample_gt = sample_gt.permute([0, 1, 3, 2])
        if args.analog_bit:
            sample_gt = bin_to_int_sample(sample_gt)
            sample = bin_to_int_sample(sample)

        save_samples(
            sample_gt, 'gt', model_kwargs, 0, num_room_types, ID_COLOR=ID_COLOR, draw_graph=args.draw_graph, save_svg=args.save_svg
        )

        save_samples(
            sample, 'pred', model_kwargs, 0, num_room_types, ID_COLOR=ID_COLOR, is_syn=True, draw_graph=args.draw_graph, save_svg=args.save_svg
        )
        
        print("Single plan done.")
        return

    # -------- ОСНОВНОЙ ЦИКЛ --------
    errors = []
    for _ in range(5):
        logger.log("sampling...")
        tmp_count = 0

        if args.dataset == 'rplan':
            ID_COLOR = {
                1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                13: '#785A67', 12: '#D3A2C7'
            }
            num_room_types = 14
            data = load_rplanhg_data(
                batch_size=args.batch_size,
                analog_bit=args.analog_bit,
                set_name=args.set_name,
                target_set=args.target_set,
            )
        else:
            print("dataset does not exist!")
            assert False
        graph_errors = []
        while tmp_count < args.num_samples:
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            data_sample, model_kwargs = next(data)
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].cpu()
            sample = sample_fn(
                model,
                data_sample.shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                analog_bit=args.analog_bit,
            )
            sample_gt = data_sample.cuda().unsqueeze(0)
            sample = sample.permute([0, 1, 3, 2])
            sample_gt = sample_gt.permute([0, 1, 3, 2])
            if args.analog_bit:
                sample_gt = bin_to_int_sample(sample_gt)
                sample = bin_to_int_sample(sample)

            graph_error = save_samples(
                sample_gt, 'gt', model_kwargs, tmp_count, num_room_types, ID_COLOR=ID_COLOR, draw_graph=args.draw_graph, save_svg=args.save_svg
            )
            graph_error = save_samples(
                sample, 'pred', model_kwargs, tmp_count, num_room_types, ID_COLOR=ID_COLOR, is_syn=True, draw_graph=args.draw_graph, save_svg=args.save_svg
            )
            graph_errors.extend(graph_error)
            tmp_count += sample_gt.shape[1]
        logger.log("sampling complete")
        fid_score = calculate_fid_given_paths(['outputs/gt', 'outputs/pred'], 64, 'cuda', 2048)
        print(f'FID: {fid_score}')
        print(f'Compatibility: {np.mean(graph_errors)}')
        errors.append([fid_score, np.mean(graph_errors)])
    errors = np.array(errors)
    print(f'Diversity mean: {errors[:, 0].mean()} \t Diversity std: {errors[:, 0].std()}')
    print(f'Compatibility mean: {errors[:, 1].mean()} \t Compatibility std: {errors[:, 1].std()}')


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        draw_graph=True,
        save_svg=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument('--single_json_file', type=str, default='', help='Path to a single json file for generation')

    return parser


if __name__ == "__main__":
    main()

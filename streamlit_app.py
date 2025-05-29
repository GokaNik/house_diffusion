import streamlit as st
import json
import os
import sys
import subprocess
import tempfile
import shutil
from PIL import Image, ImageDraw
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Streamlit UI
def main():
    st.title("HouseDiffusion Demo")
    st.write("Upload a JSON floorplan, view it and its room-graph, then generate with HouseDiffusion.")
    uploaded = st.file_uploader("Upload floorplan JSON", type=['json'])
    complete = False
    if uploaded is not None:
        # Save to temp file
        tmp_path = uploaded.name
        print(f"NAME:{tmp_path}")

        if st.button("Generate with HouseDiffusion"):
            # Clear previous outputs
            if os.path.exists('outputs'):
                shutil.rmtree('outputs')
            # Run the generation script
            cmd = [sys.executable, '-m', 'scripts.image_sample_singl',
                   '--dataset', 'rplan',
                   '--batch_size', '32',
                   '--set_name', 'eval',
                   '--target_set', '8',
                   '--single_json_file', os.path.join(r'datasets\rplan', tmp_path),
                   '--num_samples', '8',
                   '--draw_graph', 'True',
                   '--save_svg', 'False',
                   '--model_path', 'models/house_diffusion.pt']
            with st.spinner('Running HouseDiffusion...'):
                st.subheader("Generation Logs")
                log_container = st.empty()
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    log_container.text(line.rstrip())
                process.wait()
                if process.returncode != 0:
                    st.error("Generation failed, see logs above.")
                    return
            complete = True
            st.success("Generation complete!")

        while not complete:
            pass

        gt_dir = 'outputs/gt'
        gt = os.listdir(gt_dir)
        if len(gt):
            st.subheader('Ground truth')
            img_name = sorted(gt)[0]
            if img_name.endswith(('.png', '.jpg', '.svg')):
                st.image(os.path.join(gt_dir, img_name), caption=img_name, use_container_width =True)

        # Display final prediction image(s)
        pred_dir = 'outputs/pred'
        preds = os.listdir(pred_dir)
        if len(preds):
            img_name = sorted(os.listdir(pred_dir))[-1]
            if img_name.endswith(('.png', '.jpg', '.svg')):
                st.subheader(f"Generated: {img_name}")
                st.image(os.path.join(pred_dir, img_name), caption=img_name, use_container_width =True)

        gif_dir = 'outputs/gif'
        if os.path.isdir(gif_dir):
            gifs = sorted([f for f in os.listdir(gif_dir) if f.endswith('.gif')])
            if gifs:
                st.subheader('Diffusion Process GIFs')
                cols = st.columns(len(gifs))
                for col, gif in zip(cols, gifs):
                    gif_path = os.path.join(gif_dir, gif)
                    # Convert GIF to MP4 for playback controls
                    mp4_name = gif.replace('.gif', '.mp4')
                    mp4_path = os.path.join(gif_dir, mp4_name)
                    if not os.path.exists(mp4_path):
                        import imageio
                        reader = imageio.get_reader(gif_path)
                        meta = reader.get_meta_data()
                        fps = meta.get('fps', 10)
                        writer = imageio.get_writer(mp4_path, fps=fps)
                        for frame in reader:
                            writer.append_data(frame)
                        writer.close()
                        reader.close()
                    # Display as video with controls
                    col.video(mp4_path)

        # Optionally display saved graph images
        for graph_dir in ['outputs/graphs_gt', 'outputs/graphs_pred']:
            st.subheader(f"Saved graphs in {graph_dir}")
            for fname in sorted(os.listdir(graph_dir)):
                if fname.endswith(('.png', '.jpg')):
                    st.image(os.path.join(graph_dir, fname), caption=fname, use_container_width =True)


if __name__ == '__main__':
    main()

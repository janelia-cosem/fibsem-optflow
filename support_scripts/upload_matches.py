#!/usr/bin/env python

import renderapi
import argparse
import numpy as np
import os
from PIL import Image
from glob import glob


def gen_matches(flow_dir, match_name, n, stack, render_connect_params):
    render = renderapi.connect(**render_connect_params)
    tilespecs = renderapi.tilespec.get_tile_specs_from_stack(
        stack, render=render)
    spec_to_size = {tile.tileId: tile.maxY for tile in tilespecs}
    for base in glob("{}/*_top_x.tiff".format(flow_dir)):
        base = base[:-11]  # Remove the scale_top_x.tiff
        scale = np.float(base.split("_")[-1])  # Grab scale
        inv_scale = 1/scale
        base = "_".join(base.split("_")[:-1])  # Restore it
        top_bottom = ["top", "bottom"]
        base_split = base.split("/")[-1].split("~")
        groups = base_split[0].split("_")
        tiles = base_split[1:]
        w = []
        p = []
        q = []
        for s in top_bottom:
            im_x = np.array(Image.open(base+"_"+s+"_x.tiff"))
            im_y = np.array(Image.open(base+"_"+s+"_y.tiff"))

            rand = np.random.random([n, 2])*im_x.shape
            rand = rand.astype(np.int)
            w += np.ones(n).tolist()

            delta_x = np.array([im_x[tuple(j)] for j in rand])
            delta_y = np.array([im_y[tuple(j)] for j in rand])
            rand = rand.astype(np.float)

            if s == "bottom":
                rand[:, 0] += scale*spec_to_size[tiles[0]] - im_x.shape[0]
            p += (inv_scale*rand[:, [1, 0]]).tolist()
            rand[:, 1] += delta_x
            rand[:, 0] += delta_y
            q += (inv_scale*rand[:, [1, 0]]).tolist()
        p = np.array(p).transpose().tolist()
        q = np.array(q).transpose().tolist()
        matches_in = {
            'pGroupId': groups[0],
            'qGroupId': groups[1],
            'pId': tiles[0],
            'qId': tiles[1],
            'matches': {
                'p': p,
                'q': q,
                'w': w,
            }
        }
        renderapi.pointmatch.import_matches(match_name,
                                            [matches_in],
                                            render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("flow_dir", type=str)
    parser.add_argument("match", type=str)
    parser.add_argument("--n", default=25, type=int,
                        help="Number on both top & bottom")
    parser.add_argument("--stack", default="v1_acquire")
    parser.add_argument("--project", default=os.environ.get("RENDER_PROJECT"),
                        type=str)
    parser.add_argument("--owner", default=os.environ.get("RENDER_OWNER"),
                        type=str)
    parser.add_argument("--host", default=os.environ.get("RENDER_HOST"),
                        type=str)
    parser.add_argument("--port", default=os.environ.get("RENDER_PORT"),
                        type=str)
    parser.add_argument("--client_scripts",
                        default=os.environ.get("RENDER_CLIENT_SCRIPTS"),
                        type=str)
    parser.add_argument("--memGB",
                        default=os.environ.get("RENDER_CLIENT_HEAP"), type=str)
    args = parser.parse_args()

    render_connect_params = {
        "host": args.host,
        "post": args.port,
        "owner": args.owner,
        "project": args.project,
        "client_scripts": args.client_scripts,
        "memGB": args.memGB
    }

    gen_matches(args.flow_dir, args.match,
                args.n, args.stack, render_connect_params)

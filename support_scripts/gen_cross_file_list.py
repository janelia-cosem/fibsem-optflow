#!/usr/bin/env python

import renderapi
import argparse
import os
import gzip
import json


def gen_file_list(cross, stack, base_path, n, match, render_connect_params, **kwargs):

    optflow = defaults(n, **kwargs)

    optflow["host"] = render_connect_params["host"]
    optflow["port"] = render_connect_params["port"]
    optflow["matchCollection"] = match
    optflow["owner"] = render_connect_params["owner"]
    render = renderapi.connect(**render_connect_params)
    tilespecs = renderapi.tilespec.get_tile_specs_from_stack(
        stack, render=render)
    imageurls = {i.tileId: i.ip[0].imageUrl.split(":")[-1] for i in tilespecs}

    with gzip.open(cross) as f:
        pairs = json.loads(f.read().decode("ascii"))

    optflow["images"] = []
    for pair in pairs["neighborPairs"]:
        im_data = {}
        im_data["p"] = imageurls[pair["p"]["id"]]
        im_data["q"] = imageurls[pair["q"]["id"]]
        im_data["pId"] = pair["p"]["id"]
        im_data["qId"] = pair["q"]["id"]
        im_data["pGroupId"] = pair["p"]["groupId"]
        im_data["qGroupId"] = pair["q"]["groupId"]
        optflow["images"].append(im_data)
    optflow_input = ["{} {} {}_{}~{}~{}\n".format(
        imageurls[pair['p']['id']], imageurls[pair['q']['id']],
        pair['p']['groupId'], pair['q']['groupId'],
        pair['p']['id'], pair['q']['id'])
        for pair in pairs['neighborPairs']]

    with gzip.GzipFile(base_path, "w") as fout:
        fout.write(json.dumps(optflow).encode('utf-8'))


def defaults(n, **kwargs):
    d = {}
    d["style"] = kwargs.get("style", 1)
    d["debug"] = kwargs.get("debug", False)
    d["features"] = kwargs.get("features", 2)
    d["homo"] = kwargs.get("homo", 4)
    d["ratio"] = kwargs.get("ratio", 0.7)
    d["ransac"] = kwargs.get("ransac", 5)
    d["hessianThreshold"] = kwargs.get("hessianThreshold", 1600)
    d["scale"] = kwargs.get("scale", 0.5)
    d["output_dir"] = kwargs.get("output_dir", ".")
    if "top" in kwargs:
        if "rois" not in d:
            d["rois"] = {}
        if kwargs["top"]:
            d["rois"]["top"] = kwargs["top"]
    if "bottom" in kwargs:
        if "rois" not in d:
            d["rois"] = {}
        if kwargs["bottom"]:
            d["rois"]["bottom"] = kwargs["bottom"]
    d["output_type"] = kwargs.get("output_type", "random_points")
    d["npoints"] = kwargs.get("npoints", n)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cross", type=str)
    parser.add_argument("--stack", default="v1_acquire",
                        type=str)
    parser.add_argument("--base_path", default="/tmp/optflow", type=str)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--match", default="forgetful_owner", type=str)
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
    parser.add_argument("--top", default=0, type=int)
    parser.add_argument("--bottom", default=0, type=int)
    parser.add_argument("--memGB", default=os.environ.get("RENDER_CLIENT_HEAP"),
                        type=str)
    args = parser.parse_args()

    render_connect_params = {
        "host": args.host,
        "port": args.port,
        "owner": args.owner,
        "project": args.project,
        "client_scripts": args.client_scripts,
        "memGB": args.memGB
    }

    gen_file_list(args.cross, args.stack, args.base_path,
                  args.n, args.match, render_connect_params, top=args.top, bottom=args.bottom)

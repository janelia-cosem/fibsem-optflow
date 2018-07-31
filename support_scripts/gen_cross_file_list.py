#!/usr/bin/env python

import renderapi
import argparse
import os
import gzip
import json


def gen_file_list(cross, stack, base_path, n, render_connect_params):

    render = renderapi.connect(**render_connect_params)
    tilespecs = renderapi.tilespec.get_tile_specs_from_stack(
        stack, render=render)
    imageurls = {i.tileId: i.ip[0].imageUrl.split(":")[-1] for i in tilespecs}

    with gzip.open(cross) as f:
        pairs = json.loads(f.read().decode("ascii"))

    optflow_input = ["{} {} {}_{}~{}_{}\n".format(
        imageurls[pair['p']['id']], imageurls[pair['q']['id']],
        pair['p']['id'], pair['q']['id'],
        pair['p']['groupId'], pair['q']['groupId'])
        for pair in pairs['neighborpairs']]
    len_input = len(optflow_input)
    if n > 1:
        for i in range(n):
            with open("{}_{}".format(base_path, i), "w") as f:
                f.writelines(optflow_input[i*len_input//n:(i+1)*len_input//n])
    else:
        with open(base_path, "w") as f:
            f.writelines(optflow_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cross", type=str)
    parser.add_argument("--stack", default="v1_acquire",
                        type=str)
    parser.add_argument("--base_path", default="/tmp/optflow", type=str)
    parser.add_argument("--n", default=10, type=int)
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
    parser.add_argument("--memGB", default=os.environ.get("RENDER_CLIENT_HEAP"),
                        type=str)
    args = parser.parse_args()

    render_connect_params = {
        "host": args.host,
        "post": args.port,
        "owner": args.owner,
        "project": args.project,
        "client_scripts": args.client_scripts,
        "memGB": args.memGB
    }

    gen_file_list(args.cross, args.stack, args.base_path,
                  args.n, render_connect_params)

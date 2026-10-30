# Remote access

InSegt3D runs as a web server, so it can run on a remote machine with a GPU while you work in the browser on your own computer. An SSH tunnel connects a port on your computer to InSegt3D on the remote machine.

The examples use port 8080. Any free port works. Without `--port`, InSegt3D picks a free port and prints it at startup.

## A machine you can SSH into

1. On the remote machine, start InSegt3D.

   ```bash
   insegt3d --project_folder path/to/project_folder --port 8080
   ```

2. On your own computer, open a tunnel.

   ```bash
   ssh -N -L 8080:localhost:8080 username@server
   ```

   The first 8080 is the port on your computer and the second is the port InSegt3D uses.

3. Open `http://localhost:8080` in your browser.

## A cluster compute node

On most clusters you log in to a login node, and jobs run on compute nodes that only the login node can reach.

1. Start an interactive job on a compute node, ideally one with a GPU. Start InSegt3D there with `--host 0.0.0.0` so the login node can connect to it. Set a token as well, because anyone on the cluster network can reach the node.

   ```bash
   export INSEGT3D_TOKEN=choose-a-long-random-string
   insegt3d --project_folder path/to/project_folder --port 8080 --host 0.0.0.0
   ```

2. Run `hostname` on the compute node to get its name.

3. On your own computer, open a tunnel through the login node to the compute node.

   ```bash
   ssh -N -L 8080:node042:8080 username@login.cluster.example.org
   ```

4. Open `http://localhost:8080/?token=choose-a-long-random-string` in your browser.

Stop InSegt3D when you are done so the port does not stay open.

## Access token

Without a token, InSegt3D accepts every connection. Set one with `--token` or with the `INSEGT3D_TOKEN` environment variable. Other users on a shared machine can see command-line arguments, so prefer the environment variable there.

A browser or other client can supply the token in three ways.

- Add `?token=` and the token to the address. InSegt3D stores it in a cookie and removes it from the address.
- Enter it on the login page that appears when no token is given. The cookie remembers it, so you enter it once per browser.
- Send an `Authorization` header with the value `token` or `Bearer` followed by the token. This suits scripts.

The token is sent as plain text unless the connection is encrypted. An SSH tunnel encrypts the connection between your computer and the remote machine.

## Tips

- `ssh -N` holds the tunnel open until you press <kbd>Ctrl</kbd> + <kbd>C</kbd>. Add `-f` to run it in the background.
- The two ports in `-L` do not need to match. If 8080 is taken on your computer, use another port there and open that one in your browser.
- `--server_base_path` serves InSegt3D under a URL prefix, for use behind a reverse proxy.

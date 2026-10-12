# Remote access

InSegt3D runs as a web server, so it can run on a remote machine with a GPU, such as a workstation or an HPC compute node, while you use it from the browser on your own computer. An SSH tunnel connects a port on your computer to the InSegt3D server.

## Remote machine you can SSH into directly

1. SSH into the machine and start InSegt3D:

   ```bash
   insegt3d --project_folder "path/to/project_folder"
   ```

2. Note the port in the address InSegt3D prints when it starts. For example, `NiceGUI ready to go on http://localhost:37788` means port `37788`.

3. On your own computer, open a tunnel:

   ```bash
   ssh -N -L {local_port}:localhost:{insegt3d_port} {username}@{server}
   ```

4. Open `http://localhost:{local_port}` in your browser.

## HPC compute node behind a login node

On most clusters you SSH into a login node, while InSegt3D runs on a compute node that only the login node can reach.

1. Start a session on a compute node through your cluster's scheduler, preferably one with a GPU. Start InSegt3D there with `--host 0.0.0.0`, so the login node can reach it:

   ```bash
   insegt3d --project_folder "path/to/project_folder" --host 0.0.0.0
   ```

2. Note the name of the compute node (run `hostname` on it) and the port InSegt3D prints when it starts.

3. On your own computer, open a tunnel through the login node:

   ```bash
   ssh -N -L {local_port}:{compute_node}:{insegt3d_port} {username}@{login_node}
   ```

   For example, `ssh -N -L 8080:node042:37788 jeff@login.cluster.example.org`.

4. Open `http://localhost:{local_port}` in your browser.

With `--host 0.0.0.0` anyone who can reach the compute node on the cluster network can open InSegt3D, so protect it with a token as described below. Stop InSegt3D when you are done.

## Access token

By default InSegt3D asks for no login. Start it with `--token` to require a token, as Jupyter does:

```bash
insegt3d --project_folder "path/to/project_folder" --token "choose-a-long-random-string"
```

Command-line arguments are visible to other users on the machine, so on a shared system set the `INSEGT3D_TOKEN` environment variable instead.

A client then provides the token in one of three ways:

- In the address, as in `http://localhost:8080/?token=choose-a-long-random-string`. The browser stores the token in a cookie and removes it from the address.
- In an `Authorization: token choose-a-long-random-string` header (`Authorization: Bearer ...` works too).
- On the login page InSegt3D shows when neither is given. The token is stored in a cookie, so you only enter it once per browser.

The token travels unencrypted unless the connection is encrypted, as it is through an SSH tunnel.

## Tips

- `{local_port}` can be any free port on your computer, for example `8080`.
- `ssh -N` keeps the tunnel open until you press <kbd>Ctrl</kbd>+<kbd>C</kbd>. Add `-f` to run it in the background instead.
- Use `--port` to choose a fixed InSegt3D port instead of reading it from the output each time.

from multiprocessing import Process, Pipe
# from torch.multiprocessing import Process, Pipe
import gym


def worker_multi(conn, conn_send, envs):
    envs = list(envs)
    min_idx = envs[0][0]
    while True:
        cmd, datas = conn.recv()
        if cmd == "step":
            for (env_idx, env), data in zip(envs, datas):
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn_send.send((env_idx, (obs, reward, done, info)))
        elif cmd == "reset":
            for env_idx, env in envs:
                obs = env.reset()
                conn_send.send((env_idx, obs))
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.first_env = self.envs[0][0]
        self.observation_space = self.first_env.observation_space
        self.action_space = self.first_env.action_space

        self.locals = []

        self.no_envs = sum(map(len, self.envs[1:]))
        self.local_recv, remote_send = Pipe()

        env_idx = 1
        self.env_idxs = []
        for env_b in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker_multi, args=(remote, remote_send, zip(range(env_idx, env_idx+len(env_b)), env_b)))
            p.daemon = True
            p.start()
            remote.close()
            self.env_idxs.append([env_idx, env_idx+len(env_b)])
            env_idx += len(env_b)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))

        results = [self.first_env.reset()] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs

        local = self.local_recv
        while no_recv < max_recv:
            env_idx, r = local.recv()
            results[env_idx] = r
            no_recv += 1
        return results

    def step(self, actions):
        # Send Chunck actions
        for local, action_idxs in zip(self.locals, self.env_idxs):
            local.send(("step", actions[action_idxs[0]:action_idxs[1]]))
        obs, reward, done, info = self.first_env.step(actions[0])
        if done:
            obs = self.first_env.reset()

        results = [(obs, reward, done, info)] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.recv()
            results[env_idx] = r
            no_recv += 1

        results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError

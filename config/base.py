import os, sys, types

class Base(object):

    def __init__(self, cfg=None):
        if cfg is not None:
            self.update(cfg)

    def __getattr__(self, n):
        return ''

    def update(self, cfg):
        if isinstance(cfg, str):  # yaml path / dict in str
            if os.path.isfile(cfg) and cfg.endswith('yaml'):
                import yaml
                with open(cfg) as f:
                    cfg = dict(yaml.load(f, Loader=yaml.FullLoader))
            else:
                if cfg.startswith('{') and cfg.endswith('}'):
                    cfg.strip('\{\}')
                kv_list = [kv.split(':') for kv in cfg.split(',')]
                cfg = {}
                for k, v in kv_list:
                    try:
                        v = eval(v)
                    except:
                        pass
                    cfg[k] = v
            self.update(cfg)
        elif isinstance(cfg, dict):  # update from dict
            for k, v in cfg.items():
                setattr(self, k, cfg[k])
        else:  # update from another cfg
            for attr in [i for i in dir(cfg) if not i.startswith('_') and i not in self._attr_dict]:
                setattr(self, attr, getattr(cfg, attr))
        return self

    def parse(self):
        pass

    def items(self):
        for k in dir(self):
            if '__' not in k and not k.startswith('_'):
                v = getattr(self, k)
                if isinstance(v, (dict, Base)):
                    yield k, dict(v.items())
                elif not type(v) in [types.MethodType, types.FunctionType]:
                    yield k, v
    def dict(self, exclude=[]):
        return {k: v for k, v in self.items() if k not in exclude}

    def dump(self, stream=None, exclude=[], **kwargs):
        if isinstance(stream, str):
            if not stream.lower().endswith('.yaml'):
                stream = f'{stream}.yaml'
            stream = open(stream, 'w')
        import yaml
        return yaml.dump(self.dict(exclude=exclude), stream, **kwargs)

class Config(Base):
    pass
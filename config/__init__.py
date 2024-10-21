from .ft_base import AmpereConfig, SimpleConfig

def config_factory(name: str = 'ft_base'):
    if name == 'simple':
        return SimpleConfig()
    elif name == 'ampere':
        return AmpereConfig()
    else:
        raise NotImplementedError("no such config defined")

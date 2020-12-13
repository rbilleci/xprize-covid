def is_nullable(name):
    return not name.endswith('_nn')


def is_numeric(name):
    return name.endswith('_n') or name.endswith('_n_nn')


def is_boolean(name):
    return name.endswith('_b') or name.endswith('_b_nn')


def is_class(name):
    return name.endswith('_class')

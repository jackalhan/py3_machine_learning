import os
import numpy as np
import arff


def create_adff_object(name, description, element_names, classes, X_data, y_data, is_Test=False):
    obj = {}
    obj['description'] = description
    obj['relation'] = name

    _attr_list = []
    for _key, _val in element_names:
        _attr_list.append(((_val, 'REAL')))

    _attr_list.append((('class',  [_val for _key, _val in classes.items()])))
    obj['attributes'] = _attr_list
    if is_Test:
        y_data = [ "?" for x in y_data]
    try:
        obj['data'] = np.concatenate((np.array(X_data), np.array([y_data]).T), axis=1)
    except:
        obj['data'] = np.concatenate((np.array(X_data.tolist()), np.array([y_data]).T), axis=1)
    return obj


def dump_adff(name, description, element_names, classes, X_data, y_data, to_save_path, file_name, is_Test=False):
    obj = create_adff_object(name, description, element_names, classes, X_data, y_data,is_Test)
    with open(os.path.join(to_save_path, file_name + '.txt'), 'w') as outfile:
        outfile.write(arff.dumps(obj))

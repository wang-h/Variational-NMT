import os
import re
import sys
from configparser import SafeConfigParser
from Utils.utils import trace


def get_correct_args(config, items, section):
    d = {}
    for key, value in items:
        val = None
        if key == "gpu_ids":
            value = eval(value)
            if isinstance(value, list):
                val = value
            else:
                val = [int(value)]
        else:
            try:
                val = config[section].getint(key)
            except:
                try:
                    val = config[section].getfloat(key)
                except:
                    try:
                        val = config[section].getboolean(key)
                    except:
                        val = value
        d[key]= val 
    return d 


def read_config(args, args_parser, config_file=None):
    
    if config_file is None:
        return args_parser.parse_args()
    if not os.path.isfile(config_file):
        trace("""# Cannot find the configuration file. 
            {} does not exist! Please check.""".format(config_file))
        sys.exit(1)
    config = SafeConfigParser()
    config.read(config_file)
    for section in config.sections():
        default = get_correct_args(config, config.items(section), section)
        args_parser.set_defaults(**{
            k:v for k,v in filter(
                lambda x: hasattr(args, x[0]), default.items())})
    
    args = args_parser.parse_args()
    return args

def config_debug(debug):
    print(os.environ)

def format_config(args):
    ret = "\n"
    pattern = r'\<class \'(.+)\'\>'
    for key, value in vars(args).items():
        class_type = re.search(pattern, str(type(value))).group(1)
        class_type = "[{}]".format(class_type)
        value_string = str(value)
        if len(value_string) > 80:
            value_string = "/".join(value_string.split("/")[:2]) +\
                "/.../" + "/".join(value_string.split("/")[-2:])
        ret += "  {}\t{}\t{}\n".format(key.ljust(15), class_type.ljust(8), value_string)
    return ret


def config_device(config):
    if config.device_ids:
        config.device_ids = [int(x) for x in config.device_ids.split(",")]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(idx) for idx in list(
                range(config.gpu_ids, config.gpu_ids + config.num_gpus))])
        config.device_ids = list(range(config.num_gpus))

import os.path as osp
import glob
import pickle

#from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase

from yacs_utils import parsedict, parsedictlist

EXPECTED_NUM_FORWARD_SLASHES_IN_IMPATH = 2

#@DATASET_REGISTRY.register()
class DomainNetCustom(DatasetBase):
    """DomainNetCustom. Supports few-shot and class split (both from files, NOT on-the-fly from some RNG).

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    dataset_dir = "domainnet"
    domains = [
        "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, cfg, fewshot_seed, domain_split_index, class_split_type, eval_type):
        assert(domain_split_index in [0,1,2,3,4,5])
        assert(class_split_type in ['random', 'ordered'])
        assert(eval_type in ['seen_domains_seen_classes','seen_domains_unseen_classes','unseen_domains_seen_classes','unseen_domains_unseen_classes'])
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits")
        
        self._figure_out_class_stuff(cfg, self.split_dir, class_split_type, eval_type)

        self.check_input_domains(
            parsedictlist(cfg.DATASET.SOURCE_DOMAINS_LIST)[str(domain_split_index)], parsedictlist(cfg.DATASET.TARGET_DOMAINS_LIST)[str(domain_split_index)]
        )

        fewshot_filter = self._load_fewshot_filter(parsedict(cfg.DATASET.FEWSHOT_FILTER_PATHS)[str(fewshot_seed)])
        
        print('reading train_x...')
        train_x = self._read_data(parsedictlist(cfg.DATASET.SOURCE_DOMAINS_LIST)[str(domain_split_index)], self._relabeler_train, self._lab2cname_train, split="train", fewshot_filter=fewshot_filter)
        print('len(train_x)=%d'%(len(train_x)))
        
        if eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes']:
            test_input_domains = parsedictlist(cfg.DATASET.SOURCE_DOMAINS_LIST)[str(domain_split_index)]
        elif eval_type in ['unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:
            test_input_domains = parsedictlist(cfg.DATASET.TARGET_DOMAINS_LIST)[str(domain_split_index)]
        else:
            assert(False)

        print('reading test_x...')
        test = self._read_data(test_input_domains, self._relabeler_test, self._lab2cname_test, split="test")

        super().__init__(train_x=train_x, train_u=None, val=None, test=test)
        self._num_classes, self._lab2cname, self._classnames = None, None, None #these are deprecated

    def _figure_out_class_stuff(self, cfg, split_dir, class_split_type, eval_type):
        seen_class_filter, unseen_class_filter = self._load_class_filters(parsedict(cfg.DATASET.CLASS_SPLIT_PATHS)[str(class_split_type)])
        seen_relabeler = {k : v for v, k in enumerate(sorted(seen_class_filter))}
        unseen_relabeler = {k : v for v, k in enumerate(sorted(unseen_class_filter))}
        self._relabeler_train = seen_relabeler
        if eval_type in ['seen_domains_seen_classes', 'unseen_domains_seen_classes']:
            self._relabeler_test = seen_relabeler
        elif eval_type in ['seen_domains_unseen_classes', 'unseen_domains_unseen_classes']:
            self._relabeler_test = unseen_relabeler

        all_classnames_dict = {}
        txt_filenames = sorted(glob.glob(osp.join(split_dir, '*_*.txt')))
        for txt_filename in txt_filenames:
            f = open(txt_filename, 'r')
            for line in f:
                ss = line.rstrip('\n').split(' ')
                impath, label = ss
                label = int(label)
                classname = impath.split('/')[1]
                if label in all_classnames_dict:
                    assert(all_classnames_dict[label] == classname)

                all_classnames_dict[label] = classname

            f.close()

        self._lab2cname_train = {self._relabeler_train[k] : all_classnames_dict[k] for k in sorted(self._relabeler_train.keys())}
        self._lab2cname_test = {self._relabeler_test[k] : all_classnames_dict[k] for k in sorted(self._relabeler_test.keys())}
        self._classnames_train = [self._lab2cname_train[k] for k in sorted(self._lab2cname_train.keys())]
        self._classnames_test = [self._lab2cname_test[k] for k in sorted(self._lab2cname_test.keys())]
        assert(max(self._lab2cname_train.keys()) + 1 == len(self._lab2cname_train))
        assert(max(self._lab2cname_test.keys()) + 1 == len(self._lab2cname_test))
        self._num_classes_train = len(self._lab2cname_train)
        self._num_classes_test = len(self._lab2cname_test)

    def _load_class_filters(self, class_split_path):
        with open(class_split_path, 'rb') as f:
            class_filter_dict = pickle.load(f)

        return class_filter_dict['seen'], class_filter_dict['unseen']

    def _load_fewshot_filter(self, fewshot_filter_path):
        with open(fewshot_filter_path, 'rb') as f:
            fewshot_filter = pickle.load(f)

        #validate image paths - they should be exactly the same format as the text files that go into _read_data()
        for impath in sorted(fewshot_filter):
            assert(impath.count('/') == EXPECTED_NUM_FORWARD_SLASHES_IN_IMPATH)

        return fewshot_filter

    #my_relabeler acts as a class filter
    #my_lab2cname is just for double-check
    def _read_data(self, input_domains, my_relabeler, my_lab2cname, split="train", fewshot_filter=None):
        print('input_domains=%s'%(str(input_domains)))
        if fewshot_filter is not None:
            print('len(fewshot_filter)=%d'%(len(fewshot_filter)))

        items = []
        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    label = int(label)
                    assert(impath.count('/') == EXPECTED_NUM_FORWARD_SLASHES_IN_IMPATH)
                    if fewshot_filter is not None:
                        if impath not in fewshot_filter:
                            continue

                    if label not in my_relabeler:
                        continue

                    new_label = my_relabeler[label]
                    classname = impath.split("/")[1]
                    assert(classname == my_lab2cname[new_label])
                    impath = osp.join(self.dataset_dir, impath)
                    item = Datum(
                        impath=impath,
                        label=new_label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items
    
    @property
    def train_x(self):
        return self._train_x

    #NOT SUPPORTED
    @property
    def train_u(self):
        return None

    #NOT SUPPORTED
    @property
    def val(self):
        return None

    @property
    def test(self):
        return self._test

    #DEPRECATED
    @property
    def lab2cname(self):
        assert(False)
        return None

    @property
    def lab2cname_train(self):
        return self._lab2cname_train
    
    @property
    def lab2cname_test(self):
        return self._lab2cname_test

    #DEPRECATED
    @property
    def classnames(self):
        assert(False)
        return None
    
    @property
    def classnames_train(self):
        return self._classnames_train
    
    @property
    def classnames_test(self):
        return self._classnames_test

    #DEPRECATED
    @property
    def num_classes(self):
        assert(False)
        return None
    
    @property
    def num_classes_train(self):
        return self._num_classes_train
    
    @property
    def num_classes_test(self):
        return self._num_classes_test

    #DEPRECATED
    @staticmethod
    def get_num_classes(data_source):
        return None

    #DEPRECATED
    @staticmethod
    def get_lab2cname(data_source):
        return None, None

#    @staticmethod
#    def get_num_classes(data_source):
#        """Count number of classes.
#
#        Args:
#            data_source (list): a list of Datum objects.
#        """
#        label_set = set()
#        for item in data_source:
#            label_set.add(item.label)
#
#        assert(len(label_set) == max(label_set) + 1) #I mean, why wouldn't this be true?
#
#        return max(label_set) + 1
#
#    @staticmethod
#    def get_lab2cname(data_source):
#        """Get a label-to-classname mapping (dict).
#
#        Args:
#            data_source (list): a list of Datum objects.
#        """
#        container = set()
#        for item in data_source:
#            container.add((item.label, item.classname))
#
#        mapping = {label: classname for label, classname in container}
#        labels = list(mapping.keys())
#        labels.sort()
#        classnames = [mapping[label] for label in labels]
#        return mapping, classnames

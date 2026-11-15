import numpy as np
import copy
import math
from easydict import EasyDict as edict



root = edict()
cfg = root

root.prompt_tuning_path = 'config/prompt_tuning.json'

root.run_label = 'merlin'
root.gpu_ids = '0'
root.verbose = False
root.device = None
root.cfg_name = ''
root.run_tag = ''

root.seed = 180

root.model = 'ResNet32'

# Training Dynamics
root.epochs = 2
root.within_batch_update_count = 5
root.n_models = 2
root.learning_rate = 0.001
root.momentum = 0.9
root.weight_decay = 0.0001
root.batch_size_train = 512
root.batch_size_test = 100
root.input_size = 0
root.is_cifar_10 = False
root.is_cifar_100 = False
root.is_mini_imagenet = False

# Placeholders
root.timestamp = 'placeholder'
root.output_dir = 'placeholder'

# For CL
root.continual = edict()
root.continual.task = 'permuted_mnist'
root.continual.shuffle_task = False
root.continual.shuffle_datapoints = False
root.continual.rebuild_dataset = False
root.continual.n_tasks = 5
root.continual.n_class_per_task = 0
root.continual.samples_per_task = 1000
root.continual.validation_samples_per_task = 1000
root.continual.epochs = 10
root.continual.n_finetune_epochs = 10
root.continual.finetune_learning_rate = 0.001
root.continual.learning_rate = 0.001
root.continual.batch_size_train = 128
root.continual.batch_size_test = 128

root.continual.method = edict()
root.continual.method.run_merlin = True
root.continual.method.run_ewc = False
root.continual.method.run_gem = False
root.continual.method.run_icarl = False
root.continual.method.run_single_model = False
root.continual.method.run_gss = False
root.continual.method.run_gss_greedy = False

#For downstream task
root.dtask = edict()
root.dtask.type = 'class_increment'
root.dtask.nb_classes = 100
root.dtask.batch_size = 16
root.dtask.epochs = 1
root.dtask.model = 'vit_base_patch16_224'
root.dtask.input_size = 224
root.dtask.pretrained = True
root.dtask.drop = 0.0
root.dtask.drop_path = 0.0
root.dtask.opt = 'adam'
root.dtask.opt_eps = 1e-8
root.dtask.opt_betas = (0.9, 0.999)
root.dtask.clip_grad = 1.0
root.dtask.momentum = 0.9
root.dtask.weight_decay = 0.0
root.dtask.reinit_optimizer = True
root.dtask.sched = 'constant'
root.dtask.lr = 0.03
root.dtask.lr_noise = None
root.dtask.lr_noise_pct = 0.67
root.dtask.lr_noise_std = 1.0
root.dtask.warmup_lr = 1e-6
root.dtask.min_lr = 1e-5
root.dtask.decay_epochs = 30
root.dtask.warmup_epochs = 5
root.dtask.cooldown_epochs = 10
root.dtask.patience_epochs = 10
root.dtask.decay_rate = 0.1
root.dtask.unscale_lr = True
root.dtask.color_jitter = None
root.dtask.aa = None
root.dtask.smoothing = 0.1
root.dtask.train_interpolation = 'bicubic'
root.dtask.reprob = 0.0
root.dtask.remode = 'pixel'
root.dtask.recount = 1
root.dtask.data_path = './local_datasets/'
root.dtask.download = False            # dataset acquisition is an explicit preparation step
root.dtask.split_seed = 0              # fixed ImageNet-R partition, independent of stream seed
root.dtask.dataset = 'Split-CIFAR100'
root.dtask.shuffle = False
root.dtask.output_dir = './output_cifar/'
root.dtask.seed = 42
root.dtask.eval = 'store_true'
root.dtask.num_workers = 4
root.dtask.pin_mem = 'store_true'
root.dtask.no_pin_mem = 'store_false'
root.dtask.train_mask = True
root.dtask.task_inc = False
root.dtask.initializer = 'uniform'
root.dtask.use_prompt_mask = False
root.dtask.batchwise_prompt = True
root.dtask.predefined_key = ''
root.dtask.pull_constraint = True
root.dtask.pull_constraint_coeff = 0.1
root.dtask.global_pool = 'token'
root.dtask.head_type = 'prompt'
root.dtask.freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
root.dtask.print_freq = 10

#For signature learning
root.signature = edict()
root.signature.cooldown_limit = 50
root.signature.n_clusters = 8
root.signature.split_threshold = 0.2
root.signature.window_t = 20
root.signature.k = 1.2
root.signature.split_warmup = 30
root.signature.split_minpoints = 10

# Thesis improvements over the published HESTIA code (see reports/).
# Defaults reproduce the original algorithm; options marked "exact" change
# only speed/memory, not results.
root.improve = edict()
root.improve.device = 'auto'            # 'auto' (cuda > mps > cpu) | 'cuda' | 'mps' | 'cpu'
# Exact speedups (identical math to the original implementation).
root.improve.fast_head = True           # train the head on cached pre-logits instead of E full forwards
root.improve.fast_solve = True          # per-batch ridge solves via Woodbury (m x m instead of d x d)
root.improve.inmemory_bank = True       # swap IA3 vectors in one model instead of deep-copying T ViTs
# Routing / task keys.
root.improve.routing_key = 'gmm'        # 'gmm' (original online diagonal GMM) | 'class_gauss'
root.improve.routing_space = 'adapter'  # 'adapter' (z = h(x; w0 + dw_t)) | 'pretrained' (z = h(x; w0));
                                        # with 'gmm' + 'pretrained' an auxiliary GMM (same hyper-parameters)
                                        # is fit on pretrained embeddings; detection keeps the original GMM
root.improve.routing_unit = 'batch'     # 'batch' (paper protocol: one adapter per test batch) | 'sample'
root.improve.gmm_score = 'mahal_min'    # 'mahal_min' (original) | 'nll' (diagonal-GMM log-likelihood)
root.improve.gmm_var_shrink = 0.0       # pseudo-count shrinking component variances to the pooled variance
root.improve.key_view = 'aug'           # features for class_gauss keys: 'aug' (training view) | 'clean' (test transform)
root.improve.cov_type = 'full'          # class_gauss covariance: 'full' | 'diag'
root.improve.cov_scope = 'shared'       # 'shared' (pooled over all seen classes) | 'task' (pooled per adapter)
root.improve.cov_shrink_mode = 'oas'    # 'oas' (Chen et al. 2010, data-driven) | 'fixed'
root.improve.cov_shrink = 0.1           # shrinkage intensity when cov_shrink_mode == 'fixed'
root.improve.class_prior = 'empirical'  # within-task class weights: 'empirical' | 'uniform'
root.improve.fusion_alpha = 0.0         # weight of the class-Gaussian posterior fused into the head (0 = off)
# Linearized adaptation.
root.improve.select_metric = 'acc'      # soft-target refinement selection: 'acc' (original) | 'loss'
root.improve.select_tol = 0.0           # minimum improvement that resets patience (original: 0)
root.improve.refine_max_iters = 200     # cap on refinement iterations (original: 200)
root.improve.accumulate = 'merge'       # g += A^T A w_b (released code) | 'joint': g += A^T b_b (paper Eq. 5)
root.improve.tangent_solver = 'block'   # original | coupled_sketch: research candidate
root.improve.tangent_sketch_rank = 128
root.improve.tangent_residual_target = True  # used only by coupled_sketch
root.improve.prediction = 'hard'       # hard: reference; mass: support-local specialist research
root.improve.mass_inference = 'full'   # full | certified | budget | anchor | soft_mixture
root.improve.mass_alpha = 0.8          # maximum outgoing probability fraction per class
root.improve.mass_max_experts = 2      # per-image cap, only used by budget inference
root.improve.anchor_temperature = 1.0 # predeclared; never fitted on test labels

root.runtime = edict()
root.runtime.jacobian_mode = 'batch'   # reference batch Jacobian | sample-independent per_sample
root.runtime.jacobian_chunk_size = 1   # exact Jacobian rows; lower peak memory than full vectorization
root.runtime.efficient_head_loop = False  # avoid redundant per-epoch metrics and synchronizations
root.runtime.empty_cache_each_batch = True  # reference behavior; false retains the CUDA allocator cache
root.runtime.save_ridge_stats = False # evaluation needs solved adapters, not large Gram matrices
root.runtime.max_train_batches = 0    # per task; nonzero means diagnostic, never a benchmark result
root.runtime.max_eval_batches = 0     # per task
root.runtime.max_tasks = 0            # prefix of the unchanged stream, for a diagnostic pilot only
root.runtime.mixed_eval = False       # shuffle all observed validation sources before batching

_DEFAULTS = copy.deepcopy(root)


def reset_config():
    """Reset the singleton in place (also used by offline integration tests)."""
    root.clear()
    root.update(copy.deepcopy(_DEFAULTS))


def validate_config():
    """Fail before model construction for unsupported or inconsistent experiments."""
    if root.runtime.jacobian_mode not in ('batch', 'per_sample'):
        raise ValueError('runtime.jacobian_mode must be batch or per_sample')
    choices = {
        'routing_key': ('gmm', 'class_gauss'), 'routing_space': ('adapter', 'pretrained'),
        'routing_unit': ('batch', 'sample'), 'gmm_score': ('mahal_min', 'nll'),
        'key_view': ('aug', 'clean'), 'cov_type': ('full', 'diag'),
        'cov_scope': ('shared', 'task'), 'cov_shrink_mode': ('oas', 'fixed'),
        'class_prior': ('empirical', 'uniform'), 'select_metric': ('acc', 'loss'),
        'accumulate': ('merge', 'joint'), 'device': ('auto', 'cpu', 'cuda', 'mps'),
        'tangent_solver': ('block', 'coupled_sketch'),
        'prediction': ('hard', 'mass'),
        'mass_inference': ('full', 'certified', 'budget', 'anchor', 'soft_mixture'),
    }
    for key, allowed in choices.items():
        if root.improve[key] not in allowed:
            raise ValueError(f'improve.{key} must be one of {allowed}')
    for name, value in (
        ('dtask.nb_classes', root.dtask.nb_classes), ('dtask.batch_size', root.dtask.batch_size),
        ('dtask.epochs', root.dtask.epochs), ('continual.n_tasks', root.continual.n_tasks),
        ('signature.n_clusters', root.signature.n_clusters),
        ('signature.window_t', root.signature.window_t),
        ('improve.refine_max_iters', root.improve.refine_max_iters),
        ('improve.tangent_sketch_rank', root.improve.tangent_sketch_rank),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name, value in [('dtask.num_workers', root.dtask.num_workers),
                        ('dtask.split_seed', root.dtask.split_seed),
                        ('signature.cooldown_limit', root.signature.cooldown_limit),
                        *[(f'runtime.{k}', root.runtime[k]) for k in
                          ('jacobian_chunk_size', 'max_train_batches', 'max_eval_batches', 'max_tasks')]]:
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for name in ('fusion_alpha', 'gmm_var_shrink', 'select_tol', 'cov_shrink'):
        value = root.improve[name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'improve.{name} must be finite and nonnegative')
    if root.improve.cov_shrink > 1:
        raise ValueError('improve.cov_shrink must be in [0, 1]')
    imp = root.improve
    if not math.isfinite(imp.mass_alpha) or not 0 <= imp.mass_alpha < 1:
        raise ValueError('improve.mass_alpha must be in [0, 1)')
    if not math.isfinite(imp.anchor_temperature) or imp.anchor_temperature <= 0:
        raise ValueError('improve.anchor_temperature must be positive')
    if type(imp.mass_max_experts) is not int or imp.mass_max_experts < 0:
        raise ValueError('improve.mass_max_experts must be a nonnegative integer')
    if imp.prediction == 'mass' and (imp.routing_unit != 'sample' or imp.fusion_alpha):
        raise ValueError('mass prediction requires routing_unit=sample and fusion_alpha=0')
    if root.runtime.mixed_eval and imp.prediction != 'mass':
        raise ValueError('mixed_eval is currently implemented for mass prediction only')
    if imp.fusion_alpha and (imp.routing_key != 'class_gauss' or imp.routing_space != 'pretrained'):
        raise ValueError('fusion requires class_gauss keys in pretrained space')
    if imp.routing_key == 'class_gauss' and imp.routing_space == 'adapter' and imp.cov_scope == 'shared':
        raise ValueError('adapter-space class_gauss keys require cov_scope=task')
    if imp.fast_head and (root.dtask.drop or root.dtask.drop_path):
        raise ValueError('cached head training requires zero drop/drop_path; set improve.fast_head=false')
    if not imp.inmemory_bank:
        raise ValueError('only improve.inmemory_bank=true is implemented')
    if imp.tangent_solver == 'coupled_sketch' and root.runtime.save_ridge_stats:
        raise ValueError('coupled_sketch stores solved adapters; save_ridge_stats must be false')
    if root.dtask.task_inc:
        raise ValueError('this experiment runner supports global class labels, not task_inc=true')
    if not root.continual.method.run_merlin:
        raise ValueError('continual.method.run_merlin must be true to run this implementation')
    if root.run_label in ('CIFAR-10', 'CIFAR-100', 'IMAGENET-R', 'TINY-IMAGENET', 'VTAB5T'):
        if root.dtask.nb_classes % root.continual.n_tasks:
            raise ValueError('nb_classes must be divisible by n_tasks')
    if root.dtask.input_size != 224:
        raise ValueError('the registered pretrained ViT and benchmark transforms require input_size=224')

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not b.__contains__(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], tuple) and isinstance(v, list):
                v = tuple(v)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    import json
    with open(filename, 'r') as f:
        # YAML 1.1 parses JSON's 1e-06 as a string. Saved JSON must use its own parser.
        yaml_cfg = edict(json.load(f) if str(filename).endswith('.json') else yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, root)


def cfg_from_list(pairs):
    """Apply ``key.sub=value`` overrides (e.g. from ``--set``), cast to the default's type."""
    import yaml
    for pair in pairs:
        key, value = pair.split('=', 1)
        parts = key.split('.')
        node = root
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError('{} is not a valid config key'.format(key))
        old = node[parts[-1]]
        if isinstance(old, bool):
            if value.lower() not in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
                raise ValueError(f'{key}: invalid boolean {value!r}')
            new = value.lower() in ('1', 'true', 'yes', 'on')
        elif isinstance(old, int):
            new = int(value)
        elif isinstance(old, float):
            new = float(value)
        elif isinstance(old, (list, tuple)):
            new = yaml.safe_load(value)
        else:
            new = value
        node[parts[-1]] = new

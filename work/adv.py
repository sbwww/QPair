import paddle


class FGM:
    def __init__(self, model, eps=1.):
        self.model = (model.module if hasattr(model, "module") else model)
        self.eps = eps
        self.backup = {}

    # only attack embedding
    def attack(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = paddle.norm(param.grad)
                if norm and not paddle.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        for name, para in self.model.named_parameters():
            if para.stop_gradient and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (model.module if hasattr(model, "module") else model)
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = paddle.norm(param.grad)
                if norm != 0 and not paddle.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if paddle.norm(r) > self.eps:
            r = self.eps * r / paddle.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and param.grad is not None:
                param.grad = self.grad_backup[name]
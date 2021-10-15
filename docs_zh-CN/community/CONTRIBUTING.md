# 参与贡献 OpenMMLab

欢迎各种形式的贡献，包括但不限于以下内容。

- 修复（文本错误，bug)
- 新的功能和组件

## 工作流程

1. fork 并 pull 最新的 OpenMMLab 仓库 (mmclassification)
2. 签出到一个新分支（不要使用 master 分支提交 PR）
3. 进行修改并提交至 fork 出的自己的远程仓库
4. 在我们的仓库中创建一个 PR

注意：如果你计划添加一些新的功能，并引入大量改动，请尽量首先创建一个 issue 来进行讨论。

## 代码风格

### Python

我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为统一的代码风格。

我们使用下列工具来进行代码风格检查与格式化：

- [flake8](http://flake8.pycqa.org/en/latest/): 一个包含了多个代码风格检查工具的封装。
- [yapf](https://github.com/google/yapf): 一个 Python 文件的格式化工具。
- [isort](https://github.com/timothycrosley/isort): 一个对 import 进行排序的 Python 工具。
- [markdownlint](https://github.com/markdownlint/markdownlint): 一个对 markdown 文件进行格式检查与提示的工具。
- [docformatter](https://github.com/myint/docformatter): 一个 docstring 格式化工具。

yapf 和 isort 的格式设置位于 [setup.cfg](https://github.com/open-mmlab/mmclassification/blob/master/setup.cfg)

我们使用 [pre-commit hook](https://pre-commit.com/) 来保证每次提交时自动进行代
码检查和格式化，启用的功能包括 `flake8`, `yapf`, `isort`, `trailing
whitespaces`, `markdown files`, 修复 `end-of-files`, `double-quoted-strings`,
`python-encoding-pragma`, `mixed-line-ending`, 对 `requirments.txt`的排序等。
pre-commit hook 的配置文件位于 [.pre-commit-config](https://github.com/open-mmlab/mmclassification/blob/master/.pre-commit-config.yaml)

在你克隆仓库后，你需要按照如下步骤安装并初始化 pre-commit hook。

```shell
pip install -U pre-commit
```

在仓库文件夹中执行

```shell
pre-commit install
```

如果你在安装 markdownlint 的时候遇到问题，请尝试按照以下步骤安装 ruby

```shell
# 安装 rvm
curl -L https://get.rvm.io | bash -s -- --autolibs=read-fail
[[ -s "$HOME/.rvm/scripts/rvm" ]] && source "$HOME/.rvm/scripts/rvm"
rvm autolibs disable

# 安装 ruby
rvm install 2.7.1
```

或者参照 [该仓库](https://github.com/innerlee/setup) 并按照指引执行 [`zzruby.sh`](https://github.com/innerlee/setup/blob/master/zzruby.sh)

在此之后，每次提交，代码规范检查和格式化工具都将被强制执行。

```{important}
在创建 PR 之前，请确保你的代码完成了代码规范检查，并经过了 yapf 的格式化。
```

### C++ 和 CUDA

我们遵照 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

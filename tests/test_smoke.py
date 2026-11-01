import inr4d


def test_package_imports():
    # 最基本的检查确保安装后的包可以被其他角色导入。
    assert inr4d.__version__

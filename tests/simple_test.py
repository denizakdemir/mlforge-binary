"""
Simple test to verify basic Python syntax and imports
"""

def test_basic_imports():
    """Test basic Python imports and syntax."""
    print("Testing basic Python functionality...")
    
    # Test basic data structures
    data = {'test': 123, 'items': [1, 2, 3]}
    print(f"✓ Basic data structures work: {data}")
    
    # Test function definition
    def add_numbers(a, b):
        return a + b
    
    result = add_numbers(2, 3)
    print(f"✓ Function definition works: {result}")
    
    # Test class definition
    class TestClass:
        def __init__(self, value):
            self.value = value
        
        def get_value(self):
            return self.value
    
    obj = TestClass(42)
    print(f"✓ Class definition works: {obj.get_value()}")
    
    # Test imports that should be available
    try:
        import sys
        print(f"✓ sys module available")
        
        import os
        print(f"✓ os module available")
        
        import pathlib
        print(f"✓ pathlib module available")
        
        import typing
        print(f"✓ typing module available")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def test_mlforge_syntax():
    """Test that MLForge-Binary code has valid syntax."""
    print("\nTesting MLForge-Binary syntax...")
    
    try:
        # Test if we can compile the modules
        import py_compile
        
        modules_to_test = [
            'mlforge_binary/__init__.py',
            'mlforge_binary/utils.py',
            'mlforge_binary/models.py',
            'mlforge_binary/preprocessing.py',
            'mlforge_binary/evaluation.py',
            'mlforge_binary/classifier.py',
            'mlforge_binary/automl.py'
        ]
        
        for module in modules_to_test:
            try:
                py_compile.compile(module, doraise=True)
                print(f"✓ {module} has valid syntax")
            except py_compile.PyCompileError as e:
                print(f"✗ {module} has syntax error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing syntax: {e}")
        return False


def main():
    """Run all tests."""
    print("MLForge-Binary Syntax and Basic Test")
    print("=" * 40)
    
    # Test basic Python functionality
    if not test_basic_imports():
        print("\n❌ Basic Python tests failed")
        return
    
    # Test MLForge syntax
    if not test_mlforge_syntax():
        print("\n❌ MLForge-Binary syntax tests failed")
        return
    
    print("\n" + "=" * 40)
    print("✅ All syntax tests passed!")
    print("MLForge-Binary code structure is valid!")
    
    print("\nNote: To run full functionality tests, install dependencies:")
    print("pip install numpy pandas scikit-learn")


if __name__ == "__main__":
    main()
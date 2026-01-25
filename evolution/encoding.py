
import sympy
import re
import torch
import ast
import inspect

class ExpressionEncoder:
    def __init__(self, features):
        """
        features: list of string expressions to track (e.g. ['x0', 'exp(x1)'])
        """
        self.features = features
        self.feature_exprs = []
        self._parse_features()

    def _parse_features(self):
        """Pre-compile sympy expressions for the features"""
        self.feature_exprs = []
        for f_str in self.features:
            try:
                expr = sympy.sympify(f_str)
                self.feature_exprs.append(expr)
            except Exception as e:
                print(f"Warning: Could not parse feature '{f_str}': {e}")
                self.feature_exprs.append(None)

    def encode(self, individual):
        """
        Generates a fingerprint for the individual.
        Returns a list of lists of integers: [ [eq0_features], [eq1_features], ... ]
        """
        if not individual.is_compiled:
            return [[0] * len(self.features)] # Fallback

        try:
            # 1. Get Params by instantiating
            # 1. Get Params (use existing compiled model instance)
            if individual.model is None:
                return [[0] * len(self.features)]
            
            model = individual.model
            
            num_vars = model.num_vars
            
            # Map params
            param_map = {}
            for name, param in model.params.items():
                if param.numel() == 1:
                    param_map[name] = float(param.item())
                else:
                    param_map[name] = float(param.mean().item())
                    
            # 2. Extract Forward Body using AST
            code = individual.code
            try:
                tree = ast.parse(code)
            except Exception as e:
                print(f"AST Parse failed: {e}")
                return [[0] * len(self.features)] * num_vars

            forward_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'forward':
                    forward_node = node
                    break
            
            if not forward_node:
                return [[0] * len(self.features)] * num_vars
                
            # Convert body nodes back to source code
            # We can't easily use ast.unparse (Python 3.9+) if user on old env strictly?
            # But standardized env usually ok.
            # However, we need to modify the body logic slightly:
            # - Remove 'return' and assign to 'result'
            # - Parameter substitution is string based, simpler to do on string.
            
            # Let's extract the lines from original code using lineno if available, 
            # Or use ast.unparse (reliable for normalizing).
            body_code = ""
            if hasattr(ast, 'unparse'):
                # Wrap body in a dummy function call or just execute line by line?
                # We want just the statements.
                body_code = ast.unparse(forward_node.body)
            else:
                # Fallback to string slicing if ast.unparse missing
                # Should not happen in modern environments.
                return [[0] * len(self.features)] * num_vars

            # 3. Context & Mocking
            sym_vars = [sympy.Symbol(f'x{i}') for i in range(num_vars)]
            
            class MockTorch:
                def stack(self, lista, *args, **kwargs): return lista
                def sin(self, x): return sympy.sin(x)
                def cos(self, x): return sympy.cos(x)
                def tan(self, x): return sympy.tan(x)
                def exp(self, x): return sympy.exp(x)
                def log(self, x): return sympy.log(x)
                def tanh(self, x): return sympy.tanh(x)
                def sqrt(self, x): return sympy.sqrt(x)
                def abs(self, x): return sympy.Abs(x)
                def sigmoid(self, x): return 1/(1+sympy.exp(-x))
                def cat(self, x, *args, **kwargs): return x # naive
            
            mock_torch = MockTorch()
            context = {'torch': mock_torch, 'sympy': sympy}
            
            # 4. Pre-process Code String
            # Replace state accesses using Regex (AST replacement is cleaner but verbose)
            # Regex is fine on normalized unparsed code
            
            # Pattern: state[..., i] -> __sym_xi
            # Use unique prefix to avoid collision with local vars like x1
            
            for i in range(num_vars):
                # Unparsed might look like: x1 = state[:, 1]
                sym_name = f'__sym_x{i}'
                body_code = re.sub(rf'state\[\.\.\.,\s*{i}\]', sym_name, body_code)
                body_code = re.sub(rf'state\[:,\s*{i}\]', sym_name, body_code)
                # Handle potential scalar access if code does that
                body_code = re.sub(rf'state\[{i}\]', sym_name, body_code)
            
            # Map Params
            for key, val in param_map.items():
                # Robust regex: self.params [ optional_space 'key' optional_space ]
                pattern = r"self\.params\s*\[\s*['\"]" + re.escape(key) + r"['\"]\s*\]"
                body_code = re.sub(pattern, str(val), body_code)

            # Add symbols locals using the prefixed name
            for i, s in enumerate(sym_vars):
                context[f'__sym_x{i}'] = s
                
            # Replace 'return ...' with 'result = ...'
            # AST unparse keeps the 'return' keyword.
            # We can use regex to find the last return or all returns?
            # Assuming one return at end usually.
            body_code = re.sub(r'^return\s+', 'result = ', body_code, flags=re.MULTILINE)
            
            # 5. Execute
            scope = context.copy()
            
            # If code has missing imports or weird stuff, it might fail
            try:
                # print(f"DEBUG Params: {list(param_map.keys())}")
                exec(body_code, scope)
            except Exception as e:
                # print(f"Exec failed: {e}")
                # print(f"Failed Code:\n{body_code}")
                # Maybe missing variables? 
                return [[0] * len(self.features)] * num_vars
                
            result = scope.get('result')
            
            if result is None:
                 return [[0] * len(self.features)] * num_vars
                 
            # Ensure matrix format
            if not isinstance(result, (list, tuple)):
                result = [result]
                
            # If result has fewer equations than num_vars?
            # We just process what we have.
            
            matrix_fp = []
            
            for eq_idx, eq in enumerate(result):
                row_fp = []
                # Expand equation
                try:
                    expanded = sympy.expand(eq)
                except:
                    row_fp = [0] * len(self.features)
                    matrix_fp.append(row_fp)
                    continue

                for feat in self.feature_exprs:
                    if feat is None:
                        row_fp.append(0)
                        continue
                    
                    found_type = 0
                    
                    # Check coefficient of the feature
                    try:
                        coeff = expanded.coeff(feat)
                        if coeff != 0 and not coeff.free_symbols:
                            if coeff > 0:
                                found_type = 1
                            else:
                                found_type = 2
                    except:
                        pass
                        
                    row_fp.append(found_type)
                
                matrix_fp.append(row_fp)
            
            return matrix_fp
            
        except Exception as e:
            print(f"Encoding Error: {e}")
            return [[0] * len(self.features)] # Should match shape?

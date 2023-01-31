"""Matrices"""

from vectors import *
from fractions import Fraction
from polynomials import *
from roots import *

disallowed_terms = lambda s: 'import' in s or 'while' in s or 'for' in s


# Function that parses text entries into numbers
def get_value(new_entry_text_var):
    try:
        if new_entry_text_var == "":
            raise ValueError
        elif disallowed_terms(new_entry_text_var):
            raise ValueError
        try:
            value = eval(new_entry_text_var)
        except SyntaxError:
            raise ValueError
        if isinstance(value, float):
            if abs(round(value * 1e4) - value * 1e4) == 0:
                new_entry = value
            else:
                new_entry = Fraction(round(value * 1e8) / 1e8)
        else:
            new_entry = value
    except ValueError:
        raise ValueError
    return new_entry


# Create an exact copy of list but nested lists are also copied; ie. lst[0] is not lst2[0]
def copy_list(lst):
    new_copied_list = []
    for item in lst:
        if isinstance(item, list):
            item = copy_list(item)
        new_copied_list.append(item)
    return new_copied_list


# For lists containing polynomials, an alternative sum method
def poly_list_sum(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return lst[0] + poly_list_sum(lst[1:])


# Order a dictionary by keys where keys are all numbers
def order_a_dictionary(dictionary):
    new_dictionary = {}
    keys_list = list(dictionary.keys())
    ordered_keys_list = sorted(keys_list)
    for key in ordered_keys_list:
        new_dictionary[key] = dictionary[key]
    return new_dictionary


def example_matrices(start=0, end=20):
    """
    Example matrices and doctests - matrix objects are at the bottom of the file:
    """
    matrix_objects = []
    for x in range(start, end):
        name = eval(f"m{x + 1}")
        matrix_objects.append(name)
    for matrix in matrix_objects:
        print(matrix.name, "\n", matrix, "\n")
        matrix.run_all()
        print(f"Successfull test for {matrix.name}.\n")
    print("Success all test matrices.")
    # vect1 = Vector("vect1", "create", [1,0,5])
    # vect2 = Vector("vect2", "create", [0,7,3])
    # vect3 = Vector("vect3", "create", [2,4,0])
    # basis = VectorSet("V basis", [vect1, vect2, vect3])
    # basis.run_all()


set_to = True
printing_statements_operations = set_to
printing_statements_identifiers = set_to
print_statements_in_run_all = set_to


class Matrix:
    def __init__(self, identifier, how="build", matrix_lst=[], in_row=True, aug=True):
        self.name = identifier
        # Dimensions
        self.h = 0
        self.w = 0
        # Lists of vectors
        self.rows = []
        self.columns = []
        # This attribute holds an exact copy of self, but without the augmented column if self has one
        # Matrix object if augmented else None
        self.without_augmented = None
        # Matrix object representing row reduced version of self
        self.reduced_matrix = None
        # Solution to the matrix if applicable - unique = single vector / infinite = list of vectors
        self.solution = None
        # Matrix properties
        self.augm = aug
        self.is_square = None
        self.symmetric = None
        self.num_pivots = 0
        self.piv_pos = []
        self.cons = None
        self.cols_indep = None
        self.invertible = None
        self.inverted_matrix = None
        self.rank = None
        self.nullity = None
        self.det = None
        self.orthogonal = None
        self.orth_cols = None
        # Eigenvalues, eigenvector, and diagonalization
        self.char_poly = None
        self.eigenvalues = None
        self.repeated_eigenvalues = None
        self.eigenvectors = None
        self.diagonalizable = None
        self.orthogonally_diagonalizable = None
        self.diagonal_matrix_expression = None
        # SVD and basis for fundamental subspaces
        self.singular_values = None
        self.svd_exp = None
        self.fundamental_subspaces = {}
        # Upper triangularization using Schur Decomposition
        self.upper_triangular_form = []
        # Building or Creating the Matrix
        if how == "build":
            self.build_matrix()
        elif how == "create":
            self.create_matrix(matrix_lst, in_row)
        # Special all zero vectors
        self.zero_vector_col = ZeroVector(self.h)
        self.zero_vector_row = ZeroVector(self.w)

    # Filling the matrix with values
    def build_matrix(self):
        """
        Build matrix is a function that takes in user input to build the matrix.
        Considers dimensions, augmented, different input types for numbers, which
        way the numbers are filled (along rows or columns), and at the end of inputting
        asks the user if they made any errors, if so where and what the correct input is.
        """
        self.augm = bool(int(input("Is the matrix an augmented matrix? 1 (Yes) or 0 (No): ")))
        self.h = int(input("How many rows does your matrix have? "))
        self.w = int(input("How many columns does your matrix have? (Including the augmented column) "))
        x = input("Are you inputing entries along rows? t/f: ")
        if x == "t":
            for r in range(self.h):
                new_row = []
                for c in range(self.w):
                    new_entry_text = input(f"Enter the number for the {r + 1}th Row and {c + 1}th Column: ")
                    num = get_value(new_entry_text)
                    new_row.append(num)
                new_row_vector = Vector(f"Row {r + 1} for {self.name}", "create", new_row)
                self.rows.append(new_row_vector)
                print(self.rows)
            self.columns = self.transpose_r_c()
        else:
            for c in range(self.w):
                new_column = []
                for r in range(self.h):
                    new_entry_text = input(f"Enter the number for the {c + 1}th Column and {r + 1}th Row: ")
                    num = get_value(new_entry_text)
                    new_column.append(num)
                new_column_vector = Vector(f"Column {c + 1} for {self.name}", "create", new_column)
                self.columns.append(new_column_vector)
                print(self.columns)
            self.rows = self.transpose_c_r()
        print(self)
        y = input("Is the information above correct? t/f: ")
        while y == "f":
            row = int(input("What row is the error on? "))
            column = int(input("What entry in that row is the error on? "))
            correct_text = input("What is the correct entry? ")
            correct = get_value(correct_text)
            self.update_a_value(row - 1, column - 1, correct)
            print(self)
            y = input("Is the information above correct? t/f: ")
        print(f"{self.name} is the following Matrix: \n{self}")
        self.obtain_without_augemented()
        self.is_m_square()
        return self

    def create_matrix(self, matrix_lst, cond):
        """
        Create matrix is a function that creates a matrix from a given 2D list.
        Cond tells the function if the nested lists are the rows or the columns.
        """
        as_vector = all([isinstance(vec, Vector) for vec in matrix_lst])
        for i in range(len(matrix_lst) - 1):
            assert len(matrix_lst[i]) == len(matrix_lst[i - 1])
        if cond:
            self.h = len(matrix_lst)
            self.w = len(matrix_lst[0])
            for row in matrix_lst:
                if as_vector:
                    row_vector = row
                else:
                    row_vector = Vector(f"Row {matrix_lst.index(row) + 1} for {self.name}", "create", row)
                self.rows.append(row_vector)
            self.columns = self.transpose_r_c()
        else:
            self.w = len(matrix_lst)
            self.h = len(matrix_lst[0])
            for col in matrix_lst:
                if as_vector:
                    col_vector = col
                else:
                    col_vector = Vector(f"Column {matrix_lst.index(col) + 1} for {self.name}", "create", col)
                self.columns.append(col_vector)
            self.rows = self.transpose_c_r()
        self.obtain_without_augemented()
        self.is_m_square()
        return self

    def obtain_without_augemented(self):
        if self.augm:
            self.without_augmented = self.copy("column", False)

    # Square and Symmetry
    def is_m_square(self):
        # Function checks if the matrix is square and symmetric
        if self.augm:
            self.is_square = self.w - 1 == self.h
        else:
            self.is_square = self.w == self.h
        if not self.is_square:
            self.symmetric = False
            self.orthogonally_diagonalizable = False
            return
        if not self.augm and self.is_square:
            self.symmetric = self.get_row() == self.get_column()
            self.orthogonally_diagonalizable = self.symmetric
        elif self.augm and self.is_square:
            self.symmetric = self.without_augmented.get_row() == self.without_augmented.get_column()
            self.orthogonally_diagonalizable = self.symmetric

    # Run All basic matrix operations and attributes
    def run_all(self):
        print("  Row reduction: \n")
        self.row_reduce()

        print("  Pivots: \n")
        self.pivots()
        if print_statements_in_run_all:
            print(f"The number of pivots is {self.num_pivots}. \n")

        """
        Inverse operations are expensive, so they will be omitted from run_all()
        
        print("  Inverse Matrix: \n")
        self.inverted_matrix = self.invert()
        if self.inverted_matrix and print_statements_in_run_all:
            print(f"The inverse of {self.name} is \n{self.inverted_matrix}. \n")
            
        """

        print(f"  Solution: \n")
        self.solution = self.solve_matrix()
        if print_statements_in_run_all:
            print(f"The solution is {self.solution}. \n")

        print("  Determinant: \n")
        self.det = self.determinant()
        if print_statements_in_run_all:
            print(f"The determinant of {self.name} is {self.det}. \n")

        print("  Columns orthogonal: \n")
        self.orth()
        if print_statements_in_run_all:
            print(f"{self.name} is an orthogonal matrix: {self.orthogonal}. \n")

        self.eigen_and_diag()

        print("  Schur Decomposition: ")
        self.schur_decomposition()

        print("  SVD: \n")
        self.run_svd()
        if print_statements_in_run_all:
            self.print_svd_exp()
        print("Finished SVD.")

    def eigen_and_diag(self):
        print("  Eigenvalues: \n")
        self.get_eigenvalues()
        if self.eigenvalues:
            if print_statements_in_run_all:
                print(f"{self.name} has eigenvalues: {self.eigenvalues}. \n")

            print("  Eigenvectors: \n")
            self.get_eigenvectors()
            if print_statements_in_run_all:
                print(f"{self.name} has eigenvectors: {self.eigenvectors}. \n")

            print("  Diagonalization: \n")
            self.diagonal_matrix_expression = self.diagonalization()
            if self.diagonal_matrix_expression and print_statements_in_run_all:
                print(f"{self.name} has the diagonal matrix expression PDP^-1: \n")
                self.print_diag_exp()

    # Row Reduction Algorithm
    def row_reduce(self, cond=True, higher_error=False):
        """
        This algorithm performs row reduction on self
        :param cond: If the provided matrix is augmented, the algorithm also performs row reduction
        :return: None - the algorithm directly modifies the instance variables of self
        """

        # Row Reduce
        def algorithm(self, count):
            """
            Row reduces from the top row to the bottom row
            :param self: The matrix that is being row reduced - a copied version of self
            :param count: The current row number
            :return: None
            """

            # Algorithm reduces from top to bottom by obtaining pivot positions
            def algorithm2(self, count):
                """
                Algorithm2 reduces from the bottom row to the top row after all pivot positions
                have been found and are equal to 1 --> ensures that the reduced matrix has Reduced Echelon form
                :param self: The matrix that is being row reduced
                :param count: The current row number
                :return: None
                """
                if count == 0:
                    return
                count1 = 0
                current_row = self.get_row(count)
                if not current_row.zeros:
                    pivot, count1 = self.find_pivot(count)
                    if pivot != 1 and pivot != 0:
                        update_with = current_row.scale_vector(1 / pivot)
                        self.update_a_row(count, update_with)
                        current_row = self.get_row(count)
                        if printing_statements_operations:
                            print(f"{1 / pivot} R_{count + 1}")
                    change = 0
                    for i in range(count):
                        scale = -self.get_value(i, count1)
                        if abs(scale) != 0:
                            scaled_vector = current_row.scale_vector(scale)
                            sum_vector = scaled_vector + self.get_row(i)
                            self.update_a_row(i, sum_vector)
                            if printing_statements_operations:
                                print(f"R_{i + 1} + {scale} R_{count + 1}")
                            change = 1
                    if change and printing_statements_operations:
                        print(f"{self} \n")
                algorithm2(self, count - 1)

            """
            Start of algorithm
                count1 represents the column number of the pivot position of the row that is currently being reduced
            """
            # When bottom row is reduced, initiate algorithm2
            if count == self.h:
                if printing_statements_operations:
                    print("Entering algorithm2 \n")
                algorithm2(self, count - 1)
                return

            # If the row is not all zeros, enter to reduce
            if not self.get_row(count).zeros:
                pivot, count1 = self.find_pivot(count)
                # BLOCK1: Flip rows if a bottom row has a pivot in front of a top row
                if count != 0:
                    diff = self.switch_rows(count, count1)
                    # If there was a change in rows, we need to get the new value and position
                    if diff:
                        pivot, count1 = self.find_pivot(count)
                # print(f"{self} \n")
                # BLOCK2: Scale current row by the value of the pivot
                if pivot != 1 and pivot != 0:
                    updatewith = self.get_row(count).scale_vector(1 / pivot)
                    self.update_a_row(count, updatewith)
                    if printing_statements_operations:
                        print(f"1/{pivot} R_{count + 1}")
                # print(f"{self} \n")
                # BLOCK3: If all the values in a row are a multiple of the pivot, divide by the pivot.
                # From BLOCK1, it is guaranteed that rows below current row have pivot on count1 or after
                for i in range(count + 1, self.h):
                    num = abs(self.get_value(i, count1))
                    if num != 0 and num != 1 and isinstance(num, int):
                        placeholder = all([abs(self.get_value(i, x)) % num == 0 for x in range(self.w)])
                        if placeholder:
                            update_with = self.get_row(i).scale_vector(1 / num)
                            self.update_a_row(i, update_with)
                            if printing_statements_operations:
                                print(f"1/{num} R_{i + 1}")
                # print(f"{self} \n")
                # BLOCK4: If all the values in a row are equal to the values of the current row, replace row with 0s
                for i in range(self.h - 1):
                    for x in range(i + 1, self.h):
                        if self.get_row(i) == self.get_row(x):
                            self.update_a_row(x, self.zero_vector_row)
                            if printing_statements_operations:
                                print(f"R_{x + 1} - R_{i + 1}")
                # print(f"{self} \n")
                # BLOCK5: Subtract current row from all rows below it.
                """
                Because numbers are rounded floats but can be irrational, division may not give an accurate value.
                Therefore, when subtracting, values which are meant to be 0 are really small numbers. Thus, if
                any abs(number) is smaller than 1e-5, it will be taken as 0.
                """
                for i in range(count, self.h - 1):
                    current_row_inloop = self.get_row(i + 1)
                    if not self.get_row(count).zeros:
                        scale = -(current_row_inloop.get(count1))
                        if abs(scale) != 0:
                            scaled_vector = self.get_row(count).scale_vector(scale)
                            sum_vector = scaled_vector + current_row_inloop
                            self.update_a_row(i + 1, sum_vector)
                            if printing_statements_operations:
                                print(f"R_{i + 2} + {scale} R_{count + 1}")
                    values = self.get_row(i + 1).get()
                    comparisons = [(abs(val) < error) for val in values]
                    if any(comparisons):
                        for val, state in zip(values, comparisons):
                            if state:
                                self.update_a_value(i + 1, values.index(val), 0)
                if printing_statements_operations:
                    print(f"{self} \n")
            algorithm(self, count + 1)

        # Row Reduce starts here
        if self.is_row_reduced():
            print("Already reduced")
            return

        if higher_error:
            error = 1e-4
        else:
            error = 1e-5

        m = self.copy()

        if printing_statements_identifiers:
            print(f"This is the starting matrix: \n{self} \n")
        if m.get_value(0, 0) == 0:
            m.order_matrix_begin()
        # Start Algorithm
        algorithm(m, 0)
        # By this point, the reduced matrix has been calculated
        # The next nested for loop simplifies Fractions
        for x in range(m.h):
            for y in range(m.w):
                item = m.get_value(x, y)
                if isinstance(item, Fraction) and item.denominator == 1:
                    m.update_a_value(x, y, item.numerator)
        # Checks how many rows are all zeros and ensures that all zero rows
        # are at the bottom of the reduced matrix
        indices = [m.rows.index(row) for row in m.rows if not row.zeros]
        len_i = len(indices)
        if len_i != self.h:
            for x in range(len_i):
                m.update_a_row(x, m.get_row(indices[x]))
            for i in range(len_i, self.h):
                m.update_a_row(i, self.zero_vector_row)
        if printing_statements_identifiers:
            print(f"This is the resulting matrix: \n{m} \n")
        self.reduced_matrix = m
        if self.without_augmented and not self.without_augmented.is_row_reduced() and cond:
            self.without_augmented.reduced_matrix = m.copy(keep_augmented=False)
            self.without_augmented.pivots()
            self.without_augmented.orth()
        # End

    # Helper functions to row_reduce
    def order_matrix_begin(self):
        """
        Switching the first row of a matrix (whose 1st entry is 0) for the first
        row whose 1st entry is not 0 to have a pivot in the 1st row. Tries for
        a row whose first entry is 1. If none, then it flips row 1 with the first
        row to have a non-zero first entry.
        :return: None
        """
        col_1 = self.get_column(0)
        row_num = 0
        for i in range(1, col_1.len):
            if col_1.get(i) == 1:
                self.exchange_rows(0, i)
                if printing_statements_operations:
                    print(f"R_1 \leftrightarrow R_{i + 1}")
                return
            elif col_1.get(i) != 0 and not row_num:
                row_num = i
        self.exchange_rows(0, row_num)
        if printing_statements_operations:
            print(f"R_1 \leftrightarrow R_{row_num + 1}")
        return

    def switch_rows(self, c0, c1):
        """
        Switch the rows of a matrix around if a bottom row has a pivot in front of the current row
        :param c0: The current row number that is being reduced
        :param c1: The column number of the pivot on the current row
        :return: 1 if there was a row switch and 0 otherwise
        """
        num = 0
        diff = 0
        for i in range(c0 + 1, self.h):
            curr_row = self.get_row(c0 + num)
            check_row = self.get_row(i)
            assert curr_row is not check_row
            c2 = 0
            for x in check_row.get():
                if x != 0:
                    break
                c2 += 1
            if c1 > c2:
                self.exchange_rows(c0 + num, i)
                if printing_statements_operations:
                    print(f"Switched row {c0 + 1 + num} and row {i + 1}")
                num += i - c0
                diff = 1
            if c0 + num == self.h - 1:
                break
        return diff

    def find_pivot(self, count):
        """
        Finds the value of a pivot and its position on a row
        :param count: Represents the row number of the matrix
        :return: The value of the entry at the pivot and its column number
        """
        vec = self.get_row(count)
        if vec.zeros:
            count1 = vec.len - 1
            return 0, count1
        count1 = 0
        for x in vec.get():
            if x != 0:
                break
            count1 += 1
        return vec.get(count1), count1

    # Solving Matrix
    def solve_matrix(self, output=None):

        def turn_lst_into_vector(lst, width, num):
            values = list(map(lambda t: t[0], lst))
            positions = list(map(lambda t: t[1], lst))
            values_gen = iter(values)
            new_vec_lst = []
            for i in range(width):
                if i in positions:
                    new_vec_lst.append(next(values_gen))
                else:
                    new_vec_lst.append(0)
            return Vector(f"x_{num + 1}", "create", new_vec_lst)

        if not self.augm and output:
            m_lst = self.get_values_columns(self.h, self.w)
            m_lst.append(output)
            app_m = Matrix("Appended Output", "create", m_lst, False, True)
            app_m.row_reduce(higher_error=True)
            app_m.pivots()
            return app_m.solve_matrix()
        try:
            assert self.is_row_reduced()
        except AssertionError:
            print("First, you must row reduce the matrix.")
            ans = input("Reduce and proceed? y/n")
            if ans == "y":
                self.row_reduce()
                self.pivots()
                return self.solve_matrix()
            else:
                return []
        if not self.augm and not output:
            print("Cannot be solved since matrix is not augmented and no output was provided.\n")
            return []
        elif self.augm and not self.cons:
            print("The system is inconsistent; Solving with Least Squares.\n")
            return (self.least_squares(), "Least squares solution")
        elif self.augm and self.cons:
            width = self.w - 1
            num_free_vars = width - self.num_pivots
            if num_free_vars == 0:
                return [self.get_red_column(-1).copy()]
            else:
                solution_lst = []
                placeholder_dict = {}
                reduced_output = self.get_red_column(-1).get()
                red_output_gen = iter(reduced_output)
                output_coefficients = []
                piv_columns = [piv_pos[1] for piv_pos in self.piv_pos]
                piv_rows = [piv_pos[0] for piv_pos in self.piv_pos]
                generator_for_row_index = iter(piv_rows)
                for i in range(width):
                    if i in piv_columns:
                        output_coefficients.append(next(red_output_gen))
                        row_index = next(generator_for_row_index)
                        reduced_row_lst = self.get_red_row(row_index).get()[:-1]
                        for j in range(len(reduced_row_lst)):
                            if j <= i:
                                continue
                            val = -reduced_row_lst[j]
                            if val != 0:
                                if j not in placeholder_dict:
                                    placeholder_dict[j] = [(val, i)]
                                else:
                                    placeholder_dict[j].append((val, i))
                    else:
                        output_coefficients.append(0)
                        if i not in placeholder_dict:
                            placeholder_dict[i] = [(1, i)]
                        else:
                            placeholder_dict[i].append((1, i))
                ordered_p_dict = order_a_dictionary(placeholder_dict)
                if printing_statements_operations:
                    print(ordered_p_dict)
                while len(output_coefficients) < width:
                    output_coefficients.append(0)
                output_vector = Vector("Coefficients", "create", output_coefficients)
                solution_lst.append((output_vector, ""))
                for values_tuple in ordered_p_dict.items():
                    lst = values_tuple[1]
                    free_col_num = values_tuple[0]
                    vector = turn_lst_into_vector(lst, width, free_col_num)
                    solution_lst.append((vector, f"x_{free_col_num + 1}"))
                return solution_lst

    def least_squares(self):
        transpose_m = self.without_augmented.transpose()
        multiplied_m = transpose_m.mult_matrices(self.without_augmented)
        transpose_vec = transpose_m.mult_matrix_v(self.get_column(-1))
        transpose_vec_lst = transpose_vec.get()
        multiplied_m_cols = multiplied_m.get_values_columns(multiplied_m.h, multiplied_m.w)
        multiplied_m_cols.append(transpose_vec_lst)
        solving_m = Matrix("Solving", "create", multiplied_m_cols, in_row=False)
        solving_m.row_reduce()
        solving_m.pivots()
        return solving_m.solve_matrix()

    # Transpose Functions when making a new matrix object
    def transpose_r_c(self):
        # Transpose from rows to columns
        for i in range(self.h - 1):
            assert self.get_row(i).len == self.get_row(i + 1).len
        t_matrix = []
        for i in range(self.w):
            change = [self.get_value(x, i) for x in range(self.h)]
            change_vector = Vector(f"Column {i + 1} for {self.name}", "create", change)
            t_matrix.append(change_vector)
        return t_matrix

    def transpose_c_r(self):
        # Transpose from columns to rows
        for i in range(self.w - 1):
            assert self.get_column(i).len == self.get_column(i + 1).len
        t_matrix = []
        for i in range(self.h):
            change = [self.get_value2(x, i) for x in range(self.w)]
            change_vector = Vector(f"Row {i + 1} for {self.name}", "create", change)
            t_matrix.append(change_vector)
        return t_matrix

    # Transpose function when a matrix is already defined - take the columns and turn them into rows
    def transpose(self):
        assert not self.augm
        columns = self.get_values_columns(self.h, self.w)
        return Matrix("transpose", "create", columns, in_row=True, aug=False)

    # Pivot finding
    def pivots(self):
        # Function calculates the number of pivot positions of a matrix
        def find_pivot2(r_matrix, count):
            # Helper function to pivots that locates pivot positions
            # Also helps to verify a valid pivot position
            vec = r_matrix[count]
            if vec.zeros:
                count1 = vec.len
                return count1
            count1 = 0
            for i in range(count, vec.len):
                if vec.get(i) != 0:
                    break
                count1 += 1
            return count1 + count

        if not self.is_row_reduced():
            x = input("First, reduce the matrix. Would you like to reduce? y/n ")
            if x == "y":
                print("Reducing and finding columns.")
                self.row_reduce()
            else:
                print("Exiting pivot function call.")
                return
        num_pivs = 0
        pivs_pos_lst = []
        num = 1 if self.augm else 0
        if self.h >= self.w:
            length = self.h
            for x in range(self.w - num):
                piv_po = find_pivot2(self.get_red_column(), x)
                if piv_po < length:
                    num_pivs += 1
                    pivs_pos_lst.append((piv_po, x))
        elif self.h < self.w:
            length = self.w
            for x in range(self.h):
                piv_po = find_pivot2(self.get_red_row(), x)
                if piv_po < length:
                    num_pivs += 1
                    pivs_pos_lst.append((x, piv_po))
        self.num_pivots = num_pivs
        self.piv_pos = pivs_pos_lst
        self.cons = self.consistent()
        self.cols_indep = self.linear_indep_check()
        self.invertible = self.is_square and self.num_pivots == self.h and self.cols_indep
        self.rank = self.num_pivots
        self.nullity = (self.w - num) - self.num_pivots

    # Consistency, Linear Independence, and Orthogonality
    def consistent(self):
        # Function checks if the matrix is consistent regardless if augmented or not
        if self.augm:
            num_bottom_rows_all_zero = len([1 for row in self.without_augmented.get_red_row() if row.zeros])
            aug_col = self.get_red_column(-1)
            for i in range(1, num_bottom_rows_all_zero + 1):
                if aug_col.get(-i) != 0:
                    return False
            return True
        else:
            return not bool(len([1 for row in self.get_red_column() if row.zeros]))

    def linear_indep_check(self):
        # Function checks if the columns of the matrix are Linearly Independent
        if self.augm:
            return self.w - 1 == self.num_pivots
        else:
            return self.w == self.num_pivots

    def orth(self):
        # Calculate if the columns of a matrix are orthogonal
        if self.reduced_matrix and not self.cols_indep:
            self.orth_cols = False
            self.orthogonal = False
            return
        num = 1 if self.augm else 0
        dot_products = []
        norms = []
        for x in range(self.w - num):
            n = self.get_column(x).norm()
            norms.append(n)
            if x == self.w - 1 - num:
                continue
            for i in range(x + 1, self.w - num):
                op = self.get_column(x).dot_product(self.get_column(i))
                print(op)
                if isinstance(op, Root):
                    print(op.multi)
                dot_products.append(op)
        lst1 = [abs(item) <= 1e-7 for item in dot_products]
        lst2 = [abs(item - 1) <= 1e-7 for item in norms]
        if printing_statements_operations:
            print("Dot products: ", lst1, " Values: ", dot_products)
            print("Norms: ", lst2, " Values: ", norms)
        ortho = all(lst1)
        normal = all(lst2)
        self.orth_cols = ortho
        self.orthogonal = ortho and normal

    # Determinant
    def determinant(self):
        # Calculate the determinant of a matrix
        # Conditions for calculating the determinant
        if not self.is_square:
            print("The matrix is not square, so its determinant cannot be computed.\n")
            return
        if self.augm:
            copy_matrix = self.without_augmented
            return copy_matrix.determinant()

        def helper(self):
            # Recursive helper function for calculating the determinant
            # Base case: 2x2 matrix
            if self.h == 2 and self.w == 2:
                return self.get_value(0, 0) * self.get_value(1, 1) - self.get_value(0, 1) * self.get_value(1, 0)
            store_values = []
            # Separating the top row from the rest of the matrix
            top_row = self.get_row(0)
            matrix_list_without_top_row = self.get_values_rows(self.h, self.w)
            matrix_list_without_top_row.pop(0)
            # Calculation and recursive call of the determinant
            for i in range(top_row.len):
                mod_matrix_list = copy_list(matrix_list_without_top_row)
                for x in range(len(mod_matrix_list)):
                    mod_matrix_list[x].pop(i)
                mod_matrix = Matrix(f"Mod of {self.name}", "create", mod_matrix_list, aug=False)
                entry = top_row.get(i)
                recursive_call = helper(mod_matrix)
                operation = entry * recursive_call * (-1) ** i
                store_values.append(operation)
            # For eigenvalue calculation, use special sum method
            if any(map(lambda x: type(x) is Polynomial, store_values)):
                return poly_list_sum(store_values)
            # Integer sum
            return sum(store_values)

        return helper(self)

    # Null Space
    def null_space(self):
        if self.augm:
            return self.without_augmented.solve_matrix(self.zero_vector_col.get())
        else:
            return self.solve_matrix(self.zero_vector_col.get())

    # Eigenvalues and Eigenvectors
    def get_eigenvalues(self):
        # Conditions for calculating the eigenvalues of a matrix
        try:
            assert self.is_square
        except AssertionError:
            print("The matrix is not square, so its eigenvalues cannot be computed.\n")
            return
        if self.augm:
            m = self.without_augmented
        else:
            m = self
        # Creating the identity times -x matrix
        identity_x_list = m.make_diagonal_matrix_list(Polynomial([0, -1]), Polynomial([0]))
        identity_x_matrix = Matrix("id-x", "create", identity_x_list, aug=False)

        # Calculating characteristic_polynomial and finding its roots
        m2 = identity_x_matrix + m
        characteristic_polynomial = m2.determinant()
        self.char_poly = characteristic_polynomial
        if printing_statements_operations:
            print(characteristic_polynomial)
        repeated_roots_of_polynomial = characteristic_polynomial.find_roots_repeated()
        roots_of_polynomial = []
        algebraic_multiplicity = {}
        for root in repeated_roots_of_polynomial:
            if root not in roots_of_polynomial:
                roots_of_polynomial.append(root)
            if root not in algebraic_multiplicity:
                algebraic_multiplicity[root] = 1
            else:
                algebraic_multiplicity[root] += 1
        if all(list(map(lambda root: type(root) is not complex, roots_of_polynomial))):
            decreasing_order_roots = sorted(roots_of_polynomial, reverse=True)
        else:
            decreasing_order_roots = roots_of_polynomial
        self.eigenvalues = decreasing_order_roots[:]
        self.repeated_eigenvalues = algebraic_multiplicity
        if printing_statements_operations:
            print(self.eigenvalues)

    def get_eigenvectors(self):
        # Eigenvalues are required to calculate the eigenvectors
        try:
            assert self.eigenvalues
        except AssertionError:
            if not self.is_square:
                print("The matrix does not have eigenvalues since it is not square.\n")
                return
            else:
                print("The matrix is square, but it's eigenvalues haven't been computed yet.")
                ans = input("Would you like to calculate eigenvalues and eigenvectors? y/n ")
                if ans == "y":
                    self.get_eigenvalues()
                    self.get_eigenvectors()
                else:
                    return

        def update_eigenvalues_list(self, eigenvalue, geometric_mult):
            """When the eigenvectors are found, this function updates the self.eigenvalue attribute
            so that it holds a tuple with the eigenvalue and its geometric multiplicity"""
            index = self.eigenvalues.index(eigenvalue)
            self.eigenvalues[index] = (eigenvalue, geometric_mult)

        if self.augm:
            m = self.without_augmented
        else:
            m = self
        eigenspaces_solutions = []
        for eigenvalue in self.eigenvalues:
            if printing_statements_identifiers:
                print(f"{eigenvalue}-Eigenspace")
                print(f"Compute the {eigenvalue}-Eigenspace (by calculating the null space of the matrix): \n")
            # To avoid redundancy, check if an eigenvalue is 0 and just provide solution
            if self.augm and eigenvalue == 0 and self.get_column(-1).get() == self.zero_vector_col.get():
                if not self.solution:
                    eigen_matrix_solution = self.solve_matrix()
                else:
                    eigen_matrix_solution = self.solution
            # All other eigenvalues
            else:
                lst = m.make_diagonal_matrix_list(eigenvalue)
                sub_matrix = Matrix("sub matrix", "create", lst, aug=False)
                eigen_matrix = m - sub_matrix
                eigen_matrix_solution = eigen_matrix.null_space()
            try:
                if eigen_matrix_solution[0][0].zeros:
                    eigen_matrix_solution.pop(0)
            except TypeError:
                if eigen_matrix_solution[0].zeros:
                    eigen_matrix_solution.pop(0)
            eigenspace_set_evectors = []
            for eigenvector in eigen_matrix_solution:
                evector = eigenvector[0]
                eigenspace_set_evectors.append(evector)
            eigenspace_set = (eigenvalue, eigenspace_set_evectors)
            update_eigenvalues_list(self, eigenvalue, len(eigenspace_set_evectors))
            eigenspaces_solutions.append(eigenspace_set)
        self.eigenvectors = eigenspaces_solutions

    def make_diagonal_matrix_list(self, dia_val, other_val=0):
        assert self.is_square
        eval_lst = []
        num = 1 if self.augm else 0
        if type(dia_val) is list:
            assert len(dia_val) == self.h
            generator = iter(dia_val)
            cond = True
        else:
            cond = False
        for i in range(self.h):
            new_row = []
            for j in range(self.w - num):
                if i == j:
                    if cond:
                        new_row.append(next(generator))
                    else:
                        new_row.append(dia_val)
                else:
                    new_row.append(other_val)
            eval_lst.append(new_row)
        return eval_lst

    def diagonalization(self):

        def helper(self, orthogonally):
            self.diagonalizable = True
            eval_lst = []
            for eigenvalue in self.eigenvalues:
                for _ in range(eigenvalue[1]):
                    eval_lst.append(eigenvalue[0])
            diagonal_matrix_lst = self.make_diagonal_matrix_list(eval_lst)
            diagonal_matrix = Matrix("Diagonal Matrix", "create", diagonal_matrix_lst, aug=False)
            if orthogonally:
                p_matrix = self.create_orthogonal_eigen_matrix()
            else:
                p_matrix = self.create_eigen_matrix()
            p_matrix.row_reduce()
            p_matrix.pivots()
            if orthogonally:
                p_matrix_inverse = p_matrix.transpose()
            else:
                p_matrix_inverse = p_matrix.invert()
            self.diagonal_matrix_expression.extend([p_matrix, diagonal_matrix, p_matrix_inverse])
            return [p_matrix, diagonal_matrix, p_matrix_inverse]

        if isinstance(self.diagonalizable, bool):
            print(f"{self.name} is diagonalizable: {self.diagonalizable}")
            if self.diagonalizable and self.diagonal_matrix_expression != []:
                return self.diagonal_matrix_expression
            elif self.diagonalizable and self.diagonal_matrix_expression == []:
                return helper(self, self.orthogonally_diagonalizable)
            else:
                return []

        self.diagonal_matrix_expression = []
        if self.orthogonally_diagonalizable:
            return helper(self, True)
        try:
            assert self.is_square
        except AssertionError:
            print("Matrix is not diagonalizable since it is not square. Do SVD instead: \n")
            self.diagonalizable = False
            return self.run_svd()
        try:
            assert self.eigenvectors
        except AssertionError:
            return self.eigen_and_diag()
        geometric_mult_list = [eigenvalue[1] for eigenvalue in self.eigenvalues]
        total_geom_mult = sum(geometric_mult_list)
        if total_geom_mult == self.h:
            return helper(self, False)
        else:
            print("The set of all the eigenvectors do not form a basis.\n")
            self.diagonalizable = False
            return

    def create_eigen_matrix(self):
        eigenvectors_lst = self.retrieve_eigenvectors()
        eigenvectors_lst_entries = list(map(lambda v: v.get(), eigenvectors_lst))
        return Matrix(f"Matrix of Eigenvectors of {self.name}", "create", eigenvectors_lst_entries, False, False)

    def create_orthogonal_eigen_matrix(self):
        """
        This function creates an orthogonal matrix built upon the eigenvectors of self.
        Specifically, it is used when diagonalizing a symmetric matrix. The eigenspaces
        of a symmetric matrix are orthogonal to each other, so eigenvectors that belong
        to different eigenspaces are orthogonal to each other. However, if two or more
        eigenvectors belong to the same eigenspace, then it is not guaranteed that they
        are orthogonal. Thus, the function checks the geometric multiplicity of each eigenvalue
        and if it is not 1, then it performs Gram Schmidt using the VectorSet class on
        the eigenvectors that belong to that eigenspace. Finally, it normalises all the
        orthogonal eigenvectors to get an orthonormal eigenvector basis.
        """
        # Checks that all the geometric multiplicities are 1 - guarantees all eigenvectors are orthogonal; if not all 1, then we must do Gram-Schmidt
        multiplicity = [eigenval[1] for eigenval in self.eigenvalues]
        all_mults_one = all(list(map(lambda x: x == 1, multiplicity)))
        if not all_mults_one:
            lst = []
            for i in range(len(multiplicity)):
                mult = multiplicity[i]
                if mult == 1:
                    # If 1, just extend.
                    lst.extend(self.eigenvectors[i][1])
                else:
                    # If not 1, Gram Schmidt and then extend
                    evecs_lst = self.eigenvectors[i][1]
                    eigenvector_set = VectorSet("Eigenvectors", evecs_lst)
                    eigenvector_set.gram_schmidt(bypass=True)
                    placeholder_lst = eigenvector_set.get_orth()
                    lst.extend(placeholder_lst)
        else:
            lst = self.retrieve_eigenvectors()
        eigenvectors_lst_entries = list(map(lambda v: v.normalise().get(), lst))
        return Matrix(f"Matrix of normalised Eigenvectors of {self.name}", "create", eigenvectors_lst_entries, False,
                      False)

    def run_svd(self):
        if self.h == 1:
            self.fundamental_subspaces["Row Space"] = None
            self.fundamental_subspaces["Null Space"] = None
            self.fundamental_subspaces["Col Space"] = None
            self.fundamental_subspaces["Col Space Perp"] = None
            self.singular_values = []
            self.svd_exp = []
            return
        # Singular Value Decomposition
        if self.augm:
            m = self.without_augmented
        else:
            m = self
        # Step 1: orthogonally diagonalise A^T*A
        transpose_m = m.transpose()
        a_t_times_a = transpose_m * m
        if printing_statements_operations:
            print(a_t_times_a)
        print("A^T * A")
        a_t_times_a.get_eigenvalues()
        a_t_times_a.get_eigenvectors()
        print("Done")
        # V Transpose matrix - the B basis
        v_eigenvector_matrix = a_t_times_a.create_orthogonal_eigen_matrix()
        v_eigenvector_matrix.orth()
        try:
            assert v_eigenvector_matrix.orthogonal
        except AssertionError:
            v_eigenvector_matrix = v_eigenvector_matrix.orthogonalize()
            v_eigenvector_matrix.orth()
        v_transpose = v_eigenvector_matrix.transpose()
        if printing_statements_identifiers:
            print("V^T \n", v_transpose, "\n")
        assert v_transpose.is_square and v_transpose.h == m.w
        # Step 2: Singular Values - note: self.eigenvalues are already stored in decreasing order if all eigenvalues are real
        try:
            evalues = a_t_times_a.retrieve_eigenvalues()
            assert all(list(map(lambda val: not isinstance(val, complex), evalues)))
            eigenvalues_without_0 = a_t_times_a.retrieve_eigenvalues_svd()
        except AssertionError:
            self.fundamental_subspaces["Row Space"] = None
            self.fundamental_subspaces["Null Space"] = None
            self.fundamental_subspaces["Col Space"] = None
            self.fundamental_subspaces["Col Space Perp"] = None
            self.singular_values = []
            self.svd_exp = []
            print("The eigenvalues of A^T * A are complex - functionality not supported.")
            return
        singular_values = list(map(lambda x: sqrt(x), eigenvalues_without_0))
        if printing_statements_identifiers:
            print("Sing Vals: ", singular_values, "\n")
        # Step 3: Build U matrix - the C basis
        col_space = []
        c_basis_lst = []
        for i in range(len(singular_values)):
            v_vector = v_eigenvector_matrix.get_column(i)
            u_vector = (m * v_vector).scale_vector(1 / singular_values[i]).normalise()
            col_space.append(u_vector)
            u_vector_entries = u_vector.get()
            c_basis_lst.append(u_vector_entries)
        # Step 4: Extend C basis if not complete by finding Col(self)^\perp == Null(self^T), then compute Gram Schmidt.
        col_space_perp = []
        if len(c_basis_lst) < m.h:
            as_list, as_vectors = transpose_m.extend_basis_orthogonally()
            c_basis_lst.extend(as_list)
            col_space_perp = as_vectors
        u_matrix = Matrix(f"Orthogonal U Matrix for {self.name}", "create", c_basis_lst, False, False)
        u_matrix.orth()
        if printing_statements_identifiers:
            print("U \n", u_matrix, "\n")
        try:
            assert u_matrix.orthogonal
        except AssertionError:
            u_matrix = u_matrix.orthogonalize()
            u_matrix.orth()
        # Step 5: Make the Zigma matrix and put everything together
        singular_values_generator = iter(singular_values)
        z_m_lst = []
        for row in range(m.h):
            row_lst = []
            for column in range(m.w):
                if row == column:
                    try:
                        row_lst.append(next(singular_values_generator))
                    except StopIteration:
                        row_lst.append(0)
                else:
                    row_lst.append(0)
            z_m_lst.append(row_lst)
        z_matrix = Matrix(f"Zigma matrix for {self.name}", "create", z_m_lst, aug=False)
        if printing_statements_identifiers:
            print("Z \n", z_matrix, "\n")
        # Extra Step: Fundamental subspaces
        row_space = []
        for i in range(m.rank):
            v_vec = v_eigenvector_matrix.get_column(i)
            row_space.append(v_vec)
        null_space = []
        for i in range(m.rank, m.w):
            v_vec = v_eigenvector_matrix.get_column(i)
            null_space.append(v_vec)
        self.fundamental_subspaces["Row Space"] = row_space
        self.fundamental_subspaces["Null Space"] = null_space
        self.fundamental_subspaces["Col Space"] = col_space
        self.fundamental_subspaces["Col Space Perp"] = col_space_perp
        self.singular_values = singular_values
        self.svd_exp = [u_matrix, z_matrix, v_transpose]
        return

    def orthogonalize(self):
        assert not self.augm
        vs = VectorSet('ortho', self.get_column())
        vs.gram_schmidt(bypass=True)
        orthonormal = vs.get_orthn()
        return Matrix('Orthogonal matrix', 'create', orthonormal, aug=False, in_row=False)

    def extend_basis_orthogonally(self):
        """
        Extends the columns of a matrix into an orthogonal basis.
        :return: list of lists, list of vectors
        """
        extended_basis = []
        solution = self.null_space()
        try:
            if solution[0][0].zeros:
                solution.pop(0)
        except TypeError:
            if solution[0].zeros:
                solution.pop(0)
        for vector in solution:
            extended_basis.append(vector[0])
        # Gram Schmidt - returns normalised vectors
        extended_basis_vecset = VectorSet("Extended C basis", extended_basis)
        extended_basis_vecset.gram_schmidt(bypass=True)
        extended_orth_basis = extended_basis_vecset.get_orthn()
        # Append vectors as lists
        to_return_lists = []
        to_return_vectors = []
        for orth_vec in extended_orth_basis:
            to_return_lists.append(orth_vec.get())
            to_return_vectors.append(orth_vec)
        return to_return_lists, to_return_vectors

    def schur_decomposition(self):
        try:
            assert self.is_square
        except AssertionError:
            print('Matrix cannot be Upper Triangularised since it is not square.')
            return None, None

        if self.h == 1 and self.w == 1:
            return 1, self

        if not self.eigenvectors:
            self.get_eigenvalues()
            self.get_eigenvectors()

        # Get eigenvalue, eigenvector pair
        count = 0
        curr = self.eigenvectors[count]
        while isinstance(curr[0], complex) and count < len(self.eigenvectors):
            count += 1
            curr = self.eigenvectors[count]
        eigenvalue = curr[0]
        eigenvector = curr[1][0].copy().normalise()
        # Extend basis using gram schmidt
        copy_of_vec = eigenvector.get().copy()
        copy_of_vec.append(0)
        matrix_to_get_basis = Matrix('m', 'create', [copy_of_vec])
        as_list, as_vectors = matrix_to_get_basis.extend_basis_orthogonally()
        # Make orthogonal matrix Q
        as_list.insert(0, eigenvector.get())
        q_matrix = Matrix('Q', 'create', as_list, aug=False, in_row=False)
        q_tilde_matrix = q_matrix.matrix_splice((0, q_matrix.h), (1, q_matrix.w))
        # Compute Q^TAQ and unpack
        if self.augm:
            computed_matrix = q_matrix.transpose() * self.without_augmented * q_matrix
        else:
            computed_matrix = q_matrix.transpose() * self * q_matrix
        smaller_a_22 = computed_matrix.matrix_splice((1, computed_matrix.h), (1, computed_matrix.w))
        a_12_transpose = computed_matrix.matrix_splice((0, 1), (1, computed_matrix.w))
        # Recursion
        P, T_tilde = smaller_a_22.schur_decomposition()
        # Get U Matrix
        inner_u_term = (q_tilde_matrix * P).get_column()
        u_list = [eigenvector.get()]
        for vector in inner_u_term:
            u_list.append(vector.get())
        u_matrix = Matrix('U', 'create', u_list, aug=False, in_row=False)
        # Get T Matrix
        placeholder = self.zero_vector_col.copy()
        placeholder.update(0, eigenvalue)
        first_col_t = placeholder.get()
        top_row = a_12_transpose * P
        t_list = [first_col_t]
        for i in range(T_tilde.w):
            new_col = T_tilde.get_column(i).get()
            new_col.insert(0, top_row.get_column(i).get()[0])
            t_list.append(new_col)
        t_matrix = Matrix('T', 'create', t_list, aug=False, in_row=False)
        self.upper_triangular_form = [u_matrix, t_matrix]
        return u_matrix, t_matrix

    def matrix_splice(self, rows: tuple, columns: tuple):
        row_list = copy_list(self.get_row())
        to_return = []
        count = rows[0]
        while count < rows[1]:
            curr_row = row_list[count]
            new_row = []
            count1 = columns[0]
            while count1 < columns[1]:
                new_row.append(curr_row.get(count1))
                count1 += 1
            to_return.append(new_row)
            count += 1
        return Matrix('spliced matrix', 'create', to_return, aug=self.augm)

    # Matrix Operations
    def __add__(self, other):
        assert self.h == other.h
        assert self.w == other.w
        assert self.augm == other.augm
        added_matrix = []
        for i in range(self.h):
            new_vector = self.get_row(i) + other.get_row(i)
            added_matrix.append(new_vector.get())
        return Matrix(f"{self.name} + {other.name}", "create", added_matrix, True, self.augm)

    def __sub__(self, other):
        assert self.h == other.h
        assert self.w == other.w
        assert self.augm == other.augm
        sub_matrix = []
        for i in range(self.h):
            new_vector = self.get_row(i) - other.get_row(i)
            sub_matrix.append(new_vector.get())
        return Matrix(f"{self.name} - {other.name}", "create", sub_matrix, True, self.augm)

    def __eq__(self, other):
        if self.augm and other.augm:
            m = self.without_augmented
            n = other.without_augmented
            return m.get_values_rows(m.h, m.w) == n.get_values_rows(n.h, n.w)
        elif not self.augm and other.augm:
            n = other.without_augmented
            return self.get_values_rows(self.h, self.w) == n.get_values_rows(n.h, n.w)
        elif self.augm and not other.augm:
            m = self.without_augmented
            return m.get_values_rows(m.h, m.w) == other.get_values_rows(other.h, other.w)
        else:
            return self.get_values_rows(self.h, self.w) == other.get_values_rows(other.h, other.w)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.mult_matrices(other)
        elif isinstance(other, Vector):
            return self.mult_matrix_v(other)
        else:
            entries = self.get_values_rows(self.h, self.w)
            for i in range(len(entries)):
                entries[i] = list(map(lambda x: x * other, entries[i]))
            return Matrix(f"Scaled {self.name}", "create", entries, True, self.augm)

    def __rmul__(self, other):
        try:
            assert check_type(other)
            return self.__mul__(other)
        except AssertionError:
            return

    def __pow__(self, num):
        assert isinstance(num, int)
        result = self.copy()
        for i in range(1, num):
            result = result * self
        return result

    def __truediv__(self, other):
        raise TypeError

    def __rtruediv__(self, other):
        raise TypeError

    def mult_matrix_v(self, vector):
        # Matrix times a vector - Ax ==> b
        assert not self.augm
        assert vector.len == self.w
        result = []
        for i in range(self.w):
            v = self.get_column(i).scale_vector(vector.get(i))
            result.append(v)
        vec = result[0] + result[1]
        l = len(result)
        if l >= 3:
            for i in range(2, l):
                vec = vec + result[i]
        return vec

    def mult_matrices(self, other):
        # Matrix times another matrix - AB ==> C
        assert self.w == other.h
        assert not self.augm and not other.augm
        result = []
        for i in range(other.w):
            vec_column = self.mult_matrix_v(other.get_column(i))
            result.append(vec_column.get())
        return Matrix(f"{self.name} times {other.name}", "create", result, False, False)

    def invert(self):
        """
        Inverts a square matrix if the pivots() method found self to be square and full rank
        :return: Matrix object representing the inverse of self or None
        """
        try:
            assert self.invertible
        except AssertionError:
            print("The matrix is not invertible.\n")
            return
        num = 1 if self.augm else 0
        rows = self.get_values_rows(self.h, self.w - num)
        identity_lst = self.make_diagonal_matrix_list(1)
        for i in range(len(rows)):
            rows[i].extend(identity_lst[i])
        m = Matrix("Row reduce", "create", rows, aug=False)
        m.row_reduce(False)
        reduced_m = m.reduced_matrix
        cols = reduced_m.get_values_columns(reduced_m.h, reduced_m.w)
        for _ in range(m.h):
            cols.pop(0)
        return Matrix("Inverted Matrix", "create", cols, in_row=False, aug=False)

    # Str and repr methods
    def __str__(self):
        grid_lines = []
        for h in range(self.h):
            grid_line = []
            for w in range(self.w):
                if w == self.w - 1 and self.augm:
                    grid_line.append("|")
                val = self.get_value(h, w)
                val = clean_number(val)
                grid_line.append(str(val))
            grid_lines.append(' '.join(grid_line))
        return " \n".join(grid_lines)

    def __repr__(self):
        """
        The __repr__ method for the matrix class returns LaTeX code so that the website can render it.

        First, it cleans the number with the clean_number() function in roots.py by simplifying the number.
        If the value of the entry in the matrix is not a basic/built-in, then this method changes the appearance of
        the object to make it "nicer" on the website. Not doing this would probably print a fraction value that
        is way too long and would take up the entire width of the screen.
        """
        grid_lines = []
        for h in range(self.h):
            grid_line = []
            for w in range(self.w):
                if (w == self.w - 1) and self.augm:
                    grid_line.append("\\bigm|")
                val = self.get_value(h, w)
                val = clean_number(val, True)
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        grid_line.append(str(val.numerator))
                    else:
                        grid_line.append("\\frac{" + str(val.numerator) + "}{" + str(val.denominator) + "}")
                elif isinstance(val, complex):
                    if val.real == 0:
                        string = f"+{str(clean_number(val.imag, True))}i" if val.imag > 0 else f"{str(clean_number(val.imag, True))}i"
                        grid_line.append(string)
                    elif val.imag == 0:
                        grid_line.append(str(clean_number(val.real, True)))
                    else:
                        string = f"+{str(clean_number(val.imag, True))}i)" if val.imag > 0 else f"{str(clean_number(val.imag, True))}i)"
                        grid_line.append("(" + str(clean_number(val.real, True)) + string)
                elif isinstance(val, Root):
                    if check_is_int(val.rep):
                        grid_line.append(str(val))
                    else:
                        rounded = round(val.get() * 1e4) / 1e4
                        grid_line.append(str(rounded))
                else:
                    grid_line.append(str(val))
            grid_lines.append(' & '.join(grid_line))
        return " \\\\ ".join(grid_lines)

    # Get value functions
    def get_values_rows(self, height, width):
        """
        Get values in row form as a 2D list from a matrix.
        Variable height and width - helpful for removing rows or columns
        """
        m_lst = []
        for r in range(height):
            new_row = []
            for c in range(width):
                new_item = self.get_value(r, c)
                new_row.append(new_item)
            m_lst.append(new_row)
        return m_lst

    def get_values_columns(self, height, width):
        """
        Get values in column form as a 2D list from a matrix.
        Variable height and width - helpful for removing rows or columns
        :param height: Desired height - number of rows to be included
        :param width: Desired width - number of columns to be included
        :return: 2D array with entries of self in given range
        """
        m_lst = []
        for c in range(width):
            new_column = []
            for r in range(height):
                new_item = self.get_value2(c, r)
                new_column.append(new_item)
            m_lst.append(new_column)
        return m_lst

    # Get a value / row / column Functions
    def get_value(self, row, column):
        """
        :param row: row index
        :param column: column index
        :return: value at row,column
        """
        return self.rows[row].get(column)

    def get_value2(self, column, row):
        """
        :param column: column index
        :param row: row index
        :return: value at row,column
        """
        return self.columns[column].get(row)

    def get_row(self, row=None):
        """
        Gets the column of self
        :param row: integer representing row number or None
        :return: the entire list of row Vectors if row is None, otherwise row Vector number row
        """
        if isinstance(row, int):
            return self.rows[row]
        else:
            return self.rows

    def get_red_row(self, row=None):
        """
        Gets the reduced column of self using the self.reduced_matrix Matrix object
        :param row: integer representing row number or None
        :return: the entire list of reduced row Vectors if column is None, otherwise column Vector number column
        """
        if isinstance(row, int):
            return self.reduced_matrix.get_row(row)
        else:
            return self.reduced_matrix.get_row()

    def get_column(self, column=None):
        """
        Gets the column of self
        :param column: integer representing column number or None
        :return: the entire list of column Vectors if column is None, otherwise column Vector number column
        """
        if isinstance(column, int):
            return self.columns[column]
        else:
            return self.columns

    def get_red_column(self, column=None):
        """
        Gets the reduced column of self using the self.reduced_matrix Matrix object
        :param column: integer representing column number or None
        :return: the entire list of reduced column Vectors if column is None, otherwise reduced column Vector number
        column
        """
        if isinstance(column, int):
            return self.reduced_matrix.get_column(column)
        else:
            return self.reduced_matrix.get_column()

    def is_row_reduced(self):
        return not (self.reduced_matrix is None)

    def retrieve_eigenvalues(self):
        """
        Get the eigenvalues of self
        :return: list of numbers (eigenvalues)
        """
        return [eigenval[0] for eigenval in self.eigenvalues]

    def retrieve_eigenvalues_svd(self):
        """
        For SVD we want a special set of eigenvalues. Namely, we want to return the repeated set of eigenvalues
        since that is the definition of the singular value. This is where we use the algebraic multiplicity
        of each eigenvalue to return the correct amount of eigenvalues. Also, a positive operator (such as A^TA)
        has eigenvalues that are all real and non-negative, but we want to ensure that no singular value is 0, so
        we check if the eigenvalue is == 0 (and < 0 while we're at it - just in case).

        Previously, the code didn't implement this, so when running SVD on the identity matrix, there would
        only be one singular value, 1, which is incorrect.
        """
        to_return = []
        for eigenval in self.repeated_eigenvalues.keys():
            if eigenval <= 0:
                continue
            for _ in range(self.repeated_eigenvalues[eigenval]):
                to_return.append(eigenval)
        return sorted(to_return, reverse=True)

    def retrieve_eigenvectors(self):
        # Returns a 1D list whose entries are the eigenvectors
        e_lst = []
        for eigenvector in self.eigenvectors:
            vectors = eigenvector[1]
            for vec in vectors:
                e_lst.append(vec)
        return e_lst

    def get_diag_exp(self, i):
        return self.diagonal_matrix_expression[i]

    def print_diag_exp(self):
        for matrix in self.diagonal_matrix_expression:
            print(matrix)
            print("\n")

    def check_diag(self):
        """
        Returns PDP^{-1} which can be used to verify Diagonalization
        """
        p = self.get_diag_exp(0)
        d = self.get_diag_exp(1)
        pinv = self.get_diag_exp(2)
        dpinv = d.mult_matrices(pinv)
        pdpinv = p.mult_matrices(dpinv)
        pdpinv.simplify_entries()
        return pdpinv

    def get_svd_exp(self, i):
        return self.svd_exp[i]

    def print_svd_exp(self):
        for matrix in self.svd_exp:
            print(matrix)
            print("\n")

    def check_svd(self):
        """
        Returns UZV^T which can be used to verify SVD
        """
        u = self.get_svd_exp(0)
        z = self.get_svd_exp(1)
        vT = self.get_svd_exp(2)
        zvT = z.mult_matrices(vT)
        uzvT = u.mult_matrices(zvT)
        uzvT.simplify_entries()
        return uzvT

    # Update value / row / column Functions
    def update_a_value(self, row, column, value):
        self.get_row(row).update(column, value)
        self.get_column(column).update(row, value)

    def update_a_row(self, index, new_row):
        # Update row vector
        self.rows[index] = new_row
        values_list = new_row.get()
        # Update column vectors
        generate_values = iter(values_list)
        for column in self.columns:
            column.update(index, next(generate_values))

    def update_a_column(self, index, new_col):
        # Update column vector
        self.columns[index] = new_col
        values_list = new_col.get()
        # Update row vectors
        generate_values = iter(values_list)
        for row in self.rows:
            row.update(index, next(generate_values))

    def exchange_rows(self, row0, row1):
        row0vec = self.get_row(row0)
        row1vec = self.get_row(row1)
        self.update_a_row(row0, row1vec)
        self.update_a_row(row1, row0vec)

    def simplify_entries(self):
        for row in range(self.h):
            for column in range(self.w):
                value = self.get_value(row, column)
                if isinstance(value, Fraction):
                    reduced_val = value.limit_denominator()
                    self.update_a_value(row, column, reduced_val)
                elif not isinstance(value, complex):
                    rounded = round(value * 1e8) / 1e8
                    self.update_a_value(row, column, rounded)
                elif isinstance(value, complex):
                    if abs(value.imag) < 1e8:
                        value = value.real
                        rounded = round(value * 1e7) / 1e7
                        self.update_a_value(row, column, rounded)

    def copy(self, how="row", keep_augmented=True):
        """
        Make an exact copy of the self with the option to remove or keep the augmented column
        """
        num = 0
        if self.augm and not keep_augmented:
            num = 1
        bl = self.augm and keep_augmented
        if how == "row":
            matrix_list = self.get_values_rows(self.h, self.w - num)
            return Matrix(f"Copy of {self.name}", "create", matrix_list, aug=bl)
        else:
            matrix_list = self.get_values_columns(self.h, self.w - num)
            return Matrix(f"Copy of {self.name}", "create", matrix_list, in_row=False, aug=bl)


# Matrices
if __name__ == "__main__":
    m1 = Matrix("m1", "create", [[6, -11], [-11, 22], [-4, 11]], in_row=False)
    m2 = Matrix("m2", "create", [[1, 4, 6], [2, 5, 7], [3, 6, 8], [4, 7, 9]], in_row=False)
    m3 = Matrix("m3", "create", [[3, -9, -6], [-4, 12, 8], [2, -6, -4], [0, 0, 0]], False)
    m4 = Matrix("m4", "create", [[1, -2, 0], [0, 1, 2], [5, -6, 8], [2, -1, 6]], False)
    m5 = Matrix("m5", "create",
                [[1, 0, 0, 0], [-4, 0, 0, 0], [-2, 1, 0, 0], [0, 0, 0, 0], [3, 0, 1, 0], [-5, -1, -4, 0]],
                False, False)
    m6 = Matrix("m6", "create", [[1, 1, 0, -1, 2], [0, -2, 1, 1, 1], [1, 5, -2, -3, 0]], False, False)
    m7 = Matrix("m7", "create", [[4, 1, 6, 1], [0, -5, 1, -1], [1, 1, 0, -5], [3, 1, 4, -1]], False, False)
    m8 = Matrix("m8", "create", [[0, -3, -6, 4, 9], [-1, -2, -1, 3, 1], [-2, -3, 0, 3, -1], [1, 4, 5, -9, -7]])
    m9 = Matrix("m9", "create", [[0, 3, -6, 6, 4, -5], [3, -7, 8, -5, 8, 9], [3, -9, 12, -9, 6, 15]])
    m10 = Matrix("m10", "create", [[4, 8, 12, -8, 16], [0, 0, 6, 7, 15], [0, 2, 9, 3, 5], [3, 7, 0, 0, 1]])
    m11 = Matrix("m11", "create", [[0, 1, 4, -5], [1, 3, 5, -2], [3, 7, 7, 6]])
    m12 = Matrix("m12", "create", [[1, 0], [0, 1]], aug=False)
    m13 = Matrix("m13", "create", [[4, -2], [-1, 3]], aug=False)
    m14 = Matrix("m14", "create", [[2, 2, -1], [1, 3, -1], [-1, -2, 2]], aug=False)
    m15 = Matrix("m15", "create", [[1, 10, 0, 4], [0, 0, 1, 3], [0, 0, 0, 0]])
    m16 = Matrix("m16", "create", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], aug=False)
    m17 = Matrix("m17", "create",
                 [[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5], [0.5, -0.5, -0.5, 0.5], [-0.5, 0.5, -0.5, 0.5]], False,
                 False)
    m18 = Matrix("m18", "create", [[7, 0, 0], [0, 4, 0], [0, 0, 5]], False, False)
    m19 = Matrix("m19", "create", [[1, 0, 0], [0, 1, 0], [0, 0, 5]], False, False)
    m20 = Matrix("m20", "create", [[9, 0, 3], [0, 25, 0], [3, 0, 1]], False, False)
    m21 = Matrix("m20", "create", [[1, 5, 7], [0, 1, 7], [0, 0, 1]], aug=False)
    m22 = Matrix('m', 'create', [[1, +1/2*sqrt(2), +1/2*sqrt(2)], [0, 1.5, -0.5], [0, 0.5, 0.5]],
                 in_row=True, aug=False)


class VectorSet:
    def __init__(self, identifier, list_vectors):
        self.name = identifier
        # List of vectors
        self.list_v = self.vs_check(list_vectors)
        # Size of the vector set
        self.len = len(self.list_v)
        self.associated_matrix = None
        # Properties - linear independence, spanning, orthogonality, and basis?
        self.is_basis = None
        self.is_li = None
        self.spanning = None
        self.orthogonal = False
        self.orthonormal = False
        self.orthogonal_list = None
        self.orthonormal_list = None

    def vs_check(self, lst):
        """
        If the given list is a list of lists, each inner list is turned into a vector.
        """
        return_list = []
        for vec in lst:
            if isinstance(vec, Vector):
                return_list.append(vec)
            elif isinstance(vec, list):
                return_list.append(Vector('v', 'create', vec))
        return return_list

    def run_all(self):
        """
        Run all vector set operations - def_matrix() and gram_schmidt()
        """
        self.def_matrix()
        self.gram_schmidt()

    def def_matrix(self):
        """
        def_matrix is the matrix representation of a given vector set and is used
        to calculate the properties of the vector set
        """
        columned_list = [vector.get() for vector in self.list_v]
        created_matrix = Matrix(f"Matrix of {self.name}", "create", columned_list, False, False)
        created_matrix.row_reduce()
        created_matrix.pivots()
        created_matrix.orth()
        self.associated_matrix = created_matrix
        self.is_li = self.associated_matrix.cols_indep
        self.spanning = self.associated_matrix.cons
        self.is_basis = self.is_li and self.spanning
        if self.is_li:
            self.orthogonal = self.associated_matrix.orth_cols
            self.orthonormal = self.associated_matrix.orthogonal
        if self.orthogonal or self.orthonormal:
            self.orth_list = self.list_v
        print(f"{self.name} is made up of Linearly Independent vectors: {self.is_li}\n",
              f"{self.name} spans R^{self.associated_matrix.h}: {self.spanning}\n",
              f"{self.name} is a basis for R^{self.associated_matrix.h}: {self.is_basis}")

    def gram_schmidt(self, bypass=False):
        """
        The Gram Schmidt Algorithm is used to find the orthogonal basis for a
        given non-orthogonal basis.
        """
        # Conditions for running the algorithm
        if not bypass:
            if self.orthogonal:
                print(f"{self.name} is already orthogonal.")
                return
            if not self.is_li:
                print(f"{self.name} is not linearly independent, thus the algorithm cannot be run.")
                return

        print("Performing Gram Schmidt.")
        orth_basis = []

        def helper(count, stop):
            """With this helper function, we keep track of how many iterations of the Proj W formula we have
            to make and we do the actual calculations"""
            # If we reach the length of our basis in the count, then we stop.
            if count == stop:
                self.orthogonal_list = orth_basis
                self.orthonormal_list = list(map(lambda v: v.normalise(), orth_basis))
                print("Finished Gram Schmidt.")
                return None
            # When we ask for the basis vectors, we splice because we don't want to somehow
            # change the original vectors

            # The first orthogonal vector is just equal to the first basis vector
            if count == 0:
                orth_basis.append(self.get(0).copy())
            # The rest of the algorithm
            else:
                # The first value that we set our new vector equal to is the corresponding basis vector, then we
                # will subtract the projection.
                new_u = self.get(count).copy()
                # This variable will keep track of all the scaled orthogonal vectors
                ortho_sum_vector = ZeroVector(new_u.len)
                # This is the actual iterative process of adding the scaled orthogonal vectors == the projection
                for i in range(count):
                    current_vector_in_lst = orth_basis[i]
                    numerator = new_u.dot_product(current_vector_in_lst)
                    if numerator == 0:
                        continue
                    denominator = current_vector_in_lst.dot_product(current_vector_in_lst)
                    if isinstance(numerator, Root) or isinstance(denominator, Root):
                        scale = numerator / denominator
                    else:
                        scale = Fraction(numerator, denominator)
                    scaled = current_vector_in_lst.scale_vector(scale)
                    ortho_sum_vector = scaled + ortho_sum_vector
                # Appends our new orthogonal vector to our Orthogonal basis
                orth_basis.append(new_u - ortho_sum_vector)
            # Recursive call increasing the count
            return helper(count + 1, stop)

        return helper(0, self.len)

    # Get a vector
    def get(self, i=None):
        """
        Get vector in vector set
        """
        if isinstance(i, int):
            return self.list_v[i]
        else:
            return self.list_v

    def get_orth(self, i=None):
        """
        Get orthogonal vector in vector set - result from G-S
        """
        if isinstance(i, int):
            return self.orthogonal_list[i]
        else:
            return self.orthogonal_list

    def get_orthn(self, i=None):
        """
        Get orthonormal vector in vector set - result from G-S
        """
        if isinstance(i, int):
            return self.orthonormal_list[i]
        else:
            return self.orthonormal_list

    # str and repr
    def __str__(self):
        grid_lines = []
        for vec in self.list_v:
            grid_line = []
            for entry in vec.get():
                grid_line.append(str(entry))
            grid_lines.append(f'{vec.name}: ' + ' '.join(grid_line))
        return " \n".join(grid_lines)

    def __repr__(self):
        grid_lines = []
        for vec in self.list_v:
            grid_line = []
            for entry in vec.get():
                grid_line.append(str(entry))
            grid_lines.append(f'{vec.name}: ' + ' '.join(grid_line))
        return " \n".join(grid_lines)

# Space: the final frontier

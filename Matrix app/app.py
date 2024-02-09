import timeit
from flask import Flask, jsonify, render_template, render_template_string, request
import numpy as np

app = Flask(__name__)
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index1.html')
#méthode Matrice * Vecteur
@app.route('/densevecteur')
def densevecteur():
    return render_template('DenseVecteur.html')
@app.route('/infVecteur')
def infVecteur():
    return render_template('infvecteur.html')
@app.route('/supvecteur')
def supVecteur():
    return render_template('supvecteur.html')
@app.route('/demiinfvecteur')
def demiinfvecteur():
    return render_template('demiinfvecteur.html')
@app.route('/demisupvecteur')
def demisupvecteur():
    return render_template('demisupvecteur.html')

#méthode Matrice * Matrice

@app.route('/bandedemibandinf')
def bandeedemibandinf():
    return render_template('bandedemibandinf.html')
@app.route('/bandinfsup')
def bandeinfsup():
    return render_template('bandinfsup.html')
@app.route('/bandeinverse')
def bandeeinverse():
    return render_template('bandeinverse.html')
@app.route('/bandetranspose')
def bandeetranspose():
    return render_template('bandetranspose.html')

#méthode résolution sys triangulaire
@app.route('/indextriangInf')
def indexetriangInf():
    return render_template('resotriangInf.html')
@app.route('/resodemiinf')
def resoludemiinf():
    return render_template('resodemiinf.html')
@app.route('/indextriangSup')
def indexetriangSup():
    return render_template('resoTriangSup.html')
@app.route('/resodemisup')
def resoludemisup():
    return render_template('resodemisup.html')


#méthode directe
@app.route('/gauss')
def gausss():
    return render_template('gauss.html')
@app.route('/gaussbivpart')
def gausspivpart():
    return render_template('gaussAvecPivotagePartiel.html')
@app.route('/gaussj')
def gausssj():
    return render_template('gaussJordan.html')
@app.route('/lu')
def lU():
    return render_template('lu.html')
@app.route('/cholesky')
def Cholesky():
    return render_template('cholesky.html')

#méthode itérative

@app.route('/jacobi')
def Jacobi():
    return render_template('jacobi.html')
@app.route('/gs')
def Gs():
    return render_template('gaussseidel.html')

########## les fonctions 

def is_band(matrix, m):
    x = 1
    y = 1
    n = len(matrix)
    for i in range(n):
        for j in range(max(0, i - m), min(i + m + 1, n)):
            if matrix[i][j] == 0:
                x = 0

    for i in range(n):
        for j in range(n):
            if not (max(0, i - m) <= j <= min(i + m, n - 1)):
                if matrix[i][j] != 0:
                    y = 0

    if x + y == 2:
        return True
    else:
        return False  


def is_positive_definite(matrix):
    n = len(matrix)
    tolerance = 1e-8

    for i in range(1, n + 1):
        sub_matrix = np.array(matrix[:i])  # Convert to NumPy array
        sub_matrix = [row[:i] for row in sub_matrix]  # Perform list slicing
        determinant = np.linalg.det(sub_matrix)

        if determinant <= tolerance:
            return False

    return True

def is_upper_triangular(A):
    rows, cols = A.shape
    for i in range(rows - 1):
        for j in range(i ):
            if A[i, j] != 0:
                return False
    return True

def is_lower_triangular(A):
    rows, cols = A.shape
    for i in range(1, rows):
        for j in range(i, cols):
            if A[i, j] != 0:
                return False
    return True

def is_lower(matrix, m):
    x = 1
    y = 1
    n = len(matrix)
    for i in range(n):
        for j in range(max(0, i - m), i + 1):
            if matrix[i][j] == 0:
                x = 0

    for i in range(n):
        for j in range(n):
            if j < max(0, i - m) or j >= i + 1:
                if matrix[i][j] != 0:
                    y = 0

    if x + y == 2:
        return True
    else:
        return False

def is_upper(matrix, m):
    x = 1
    y = 1
    n = len(matrix)
    for i in range(n):
        for j in range(i, min(i + m + 1, n)):
            if matrix[i][j] == 0:
                x = 0

    for i in range(n):
        for j in range(n):
            if i > j or j >= min(i + m + 1, n):
                if matrix[i][j] != 0:
                    y = 0
    print(x, y)
    if x + y == 2:
        return True
    else:
        return False

def check_symmetry(matrix):
    if not (np.array(matrix) == np.array(matrix).T).all():
        raise ValueError("Error: La matrice n'est pas symétrique.")
    return True, None

def check_non_symmetry(matrix):
    if  (np.array(matrix) == np.array(matrix).T).all():
        raise ValueError("Error: La matrice est pas symétrique.")
    return True, None

def find_bandwidth(matrix):
    n = len(matrix)
    m = 0  # Initialize m to 0
    for i in range(n):
        for j in range(n):
            # Check elements outside the diagonal band
            if i != j and abs(i - j) > m and matrix[i][j] != 0:
                m = abs(i - j)
    if (m <= (n-1)// 2):
       return m
    else:
         raise ValueError("Matrice n'est pas bande.")


@app.route('/menu_scalaire')
def menu_scalaire():
    return render_template('menuMatriceXScalaire.html')

# ... (your Flask code above)

@app.route('/multiply_scalar', methods=['POST'])
def multiply_scalar():
    if request.method == 'POST':
        try:
            # Get value for scalar
            scalar = float(request.form['scalar'])

            # Get values for Matrix X
            rows_x = int(request.form['rows'])
            columns_x = int(request.form['columns'])
            matrix_x = [[float(request.form[f'matrixX[{i}][{j}]']) for j in range(columns_x)] for i in range(rows_x)]

            # Perform matrix-scalar multiplication
            result = multiply_scalar_function(matrix_x, scalar)

            # Pass the result and input matrix to the template along with rows and columns
            return render_template('resultMatxSca.html', matrix_x=matrix_x, scalar=scalar, result=result, rows=rows_x, columns=columns_x)

        except ValueError:
            return "Please enter valid numeric values for matrix elements and scalar."




# ...

def multiply_scalar_function(matrix_x, scalar):
    # Perform matrix-scalar multiplication
    rows_x, cols_x = len(matrix_x), len(matrix_x[0])

    result = [[0 for _ in range(cols_x)] for _ in range(rows_x)]

    for i in range(rows_x):
        for j in range(cols_x):
            result[i][j] = matrix_x[i][j] * scalar

    return result



######### gauss jordan 

def gauss_jordan_elimination(matrix, n,b):
    cost = 0
    matrix = np.concatenate((matrix, np.expand_dims(b, axis=1)), axis=1)
    # Élimination
    for k in range(n):
        pivot = matrix[k][k]

        # Normaliser la ligne k
        for j in range(k, n + 1):
            matrix[k][j] /= pivot
        cost += 1

        # Appliquer l'élimination aux autres lignes
        for i in range(n):
            if i != k:
                factor = matrix[i][k]
                for j in range(k, n + 1):
                    matrix[i][j] -= factor * matrix[k][j]
                cost += n - k

    # Correction : Mettre à zéro les éléments au-dessus de la diagonale
    for k in range(n - 1, 0, -1):
        for i in range(k - 1, -1, -1):
            factor = matrix[i][k]
            for j in range(k, n + 1):
                matrix[i][j] -= factor * matrix[k][j]
            cost += n - k

    return matrix
def gauss_jordan_elimination_banded(matrix, n, m):
    for k in range(n):
        pivot = matrix[k][k]

        # Normaliser la ligne k
        for j in range(max(0, k - m), min(n, k + m) ):
            matrix[k][j] /= pivot
            print(k,j,matrix[k][j])
            
        # Appliquer l'élimination aux autres lignes
        for i in range(n):
            if i != k:
                factor = matrix[i][k]
                for j in range(max(0, k - m), min(n, k + m) ):
                    matrix[i][j] -= factor * matrix[k][j]
                    print(i,j,matrix[i][j])

    return matrix

def solve_jordan(band_matrix, vector, m):
    n = len(vector)
    augmented_matrix = np.zeros((n, n + 1))
    
    # Création de la matrice augmentée
    for i in range(n):
        for j in range(max(i - m, 0), min(m + i+1, n)):
            augmented_matrix[i, j] = band_matrix[i][j]
        augmented_matrix[i, n] = vector[i]

    # Élimination de Gauss-Jordan
    for k in range(n):
        for j in range(max(k + 1, k - m), min(m + k+1, n)):
            augmented_matrix[k, j] = augmented_matrix[k, j] / augmented_matrix[k, k]

        augmented_matrix[k, n] = augmented_matrix[k, n] / augmented_matrix[k, k]
        augmented_matrix[k, k] = 1

        for i in range(n):
            if i != k:
                for j in range(max(k + 1, i - m), n + 1):
                    augmented_matrix[i, j] -= augmented_matrix[i, k] * augmented_matrix[k, j]

    # Extraction de la solution
    solution = augmented_matrix[:, n]

    return solution
@app.route('/gaussj')
def indexgj():
     return render_template('gaussJordan.html')

@app.route('/eliminate', methods=['POST'])
def eliminate():
    try:
        size = int(request.form['size'])
        matrix = [[float(request.form[f'matrix[{i}][{j}]']) for j in range(size )] for i in range(size)]
        matrix_type = request.form.get('matrixType')
        vector = np.array([float(request.form[f'vector[{i}]']) for i in range(size)])
        if not is_positive_definite(matrix):
            raise ValueError("Matrice n'est pas definie positive.")
        is_symmetric = check_symmetry(matrix)
        if matrix_type == 'dense':
            result_matrix = gauss_jordan_elimination(matrix, size,vector)
        elif matrix_type == 'bande':
            m = find_bandwidth(matrix)
            if is_band(np.array(matrix),m):
                solution = solve_jordan(matrix, vector, m)
                A = gauss_jordan_elimination_banded(matrix,size,m)
                print(A)
                result_matrix = np.concatenate((A, np.expand_dims(solution, axis=1)), axis=1)
                return render_template_string("""
            <div>
                <h2>La matrice aprés l'application de Gauss Jordan:</h2>
                <table class="table table-bordered">
                    <h3>Valeur de m: {{ m }}</h3>
                    {% for row in result_matrix %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>

            </div>
        """, result_matrix=result_matrix,m=m)

        return render_template_string("""
            <div >
                <h2>La matrice aprés l'application de Gauss Jordan:</h2>
                <table class="table table-bordered">
                    {% for row in result_matrix %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                
            </div>
        """, result_matrix=result_matrix)
    except ValueError as e:
        return f"""
        <div ">
            <h2>Error:</h2>
            <p>{str(e)}</p>
            
        </div>
        """

################### LU 
@app.route('/lu')
def indexlu():
    return render_template('lu.html')
@app.route('/decompose', methods=['POST'])
def decompose():
    try:
        dimension = int(request.form['dimension'])
        matrix = [[float(request.form[f'matrix[{i}][{j}]']) for j in range(dimension)] for i in range(dimension)]
        vector = np.array([float(request.form[f'vector[{i}]']) for i in range(dimension)])
        matrix_type = request.form.get('matrixType')
        print("Matrix Type:", matrix_type)

        if not is_positive_definite(matrix):
            raise ValueError("Matrice n'est pas definie positive.")
        is_symmetric = check_symmetry(matrix)


        if matrix_type == 'dense':
            L, U,x = LU(np.array(matrix),vector)
        elif matrix_type == 'bande':
            m = find_bandwidth(matrix)
            if is_band(np.array(matrix),m):
                L, U,x = lu_decomposition_bande(np.array(matrix), dimension, m,vector)
                return render_template_string("""
            <div >
                <h3>La valeur de m :  {{ m }}</h3>
                <h2>La matrice initiale : </h2>
                <table class="table table-bordered">
                    {% for row in matrix %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                <h2>La matrice triangulaire inférieure L:</h2>
                <table class="table table-bordered">
                    {% for row in L %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                <h2>La matrice triangulaire supérieure U:</h2>
                <table class="table table-bordered">
                    {% for row in U %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>                               
                               <h2>Solution (x):</h2>
                    <table class="table table-bordered">
                    <tr>
                        {% for elem in x %}
                            <td>{{ elem }}</td>
                        {% endfor %}
                    </tr>
                </table>
                                   
                
            </div>
        """, matrix=matrix, L=L, U=U,m=m,x=x)
            else:
                raise ValueError("Matrice n'est pas bande.")
        else:
            raise ValueError("Type de matrice invalide.")   

        return render_template_string("""
            <div >
                
                <h2>La matrice initiale : </h2>
                <table class="table table-bordered">
                    {% for row in matrix %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                <h2>La matrice triangulaire inférieure L:</h2>
                <table class="table table-bordered">
                    {% for row in L %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                <h2>La matrice triangulaire supérieure U:</h2>
                <table class="table table-bordered">
                    {% for row in U %}
                        <tr>
                            {% for elem in row %}
                                <td>{{ elem }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                                       <h2>Solution(x):</h2>
                               <table class="table table-bordered">
                    <tr>
                        {% for elem in x %}
                            <td>{{ elem }}</td>
                        {% endfor %}
                    </tr>
                </table>
                                   
               
            </div>
        """, matrix=matrix, L=L, U=U,x=x)
    except ValueError as e:
        return f"""
        <div >
            <h2>Error:</h2>
            <p>{str(e)}</p>
            
        </div>
        """
    except ValueError as e:
        return render_template_string('index.html', error=str(e))


def resolution_inf_1(M, x, b):
    n = len(M)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= M[i][j] * x[j]



def LU(M,b):
    dim = len(M)
    L = np.zeros((dim, dim))
    U = np.zeros((dim, dim))
    x = np.zeros(dim)
    y = np.zeros(dim)


    cost = 0

    for i in range(dim):
        L[i][i] = 1
        for j in range(i):
            L[i][j] = M[i][j]
            for k in range(j):
                L[i][j] -= L[i][k] * U[k][j]
                cost += 2
            L[i][j] /= U[j][j]
            cost += 1

        for j in range(i, dim):
            U[i][j] = M[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
                cost += 2
  
    resolution_inf_1(L, y, b)
    resolution_sup(U, x, y)
    
    return L, U, x

def resolution_sup(U, x, y):
    n = len(U)

    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for k in range(i + 1, n):
            x[i] -= U[i][k] * x[k]
        x[i] /= U[i][i]



def lu_decomposition_bande(M, rows, m,b):
    columns = M.shape[1]
    if rows != columns:
        raise ValueError("La décomposition LU nécessite une matrice carrée")

    L = np.eye(rows)
    U = np.zeros_like(M)
    x = np.zeros(rows)
    y = np.zeros(rows)

    for i in range(rows):
        for j in range(i):
            sum_l = 0
            for k in range(max(0, i - m), min(columns - 1, i + m) + 1):
                sum_l += L[i, k] * U[k, j]
            L[i, j] = M[i, j] - sum_l
            L[i, j] /= U[j, j]

        for j in range(i, columns):
            sum_u = 0
            for k in range(max(0, i - m), min(columns - 1, i + m) + 1):
                sum_u += L[i, k] * U[k, j]
            U[i, j] = M[i, j] - sum_u
    resolution_inf_1(L, y, b)
    resolution_sup(U, x, y)

    return L, U,x

#####gauss 


@app.route('/gauss')
def indexgauss():
    return render_template('gauss.html')

@app.route('/appgauss', methods=['POST'])
def appgauss():
    try:
        # Get values for Matrix
        n = int(request.form['matrixSize'])
        matrix = np.array([[float(request.form[f'matrix[{i}][{j}]']) for j in range(n)] for i in range(n)])       
        matrix_type = request.form.get('matrixType')  # 'dense' or 'bande'
        vector = np.array([float(request.form[f'vector[{i}]']) for i in range(n)])

        if not is_positive_definite(matrix):
            raise ValueError("La matrice n'est pas définie positive")
        # Check if the matrix is symmetric
        is_symmetric = check_symmetry(matrix)

            
        if matrix_type == 'dense':
            processed_matrix,x = process_matrix(matrix,vector)

        elif matrix_type == 'bande':
            m = find_bandwidth(matrix)
            if is_band(matrix,m):
               
                processed_matrix,x = process_matrixBande(matrix, vector,m)
            else:
                raise ValueError("Matrice n'est pas bande.")
        else:
            raise ValueError("Type de matrice invalide.")
      
        m_value = find_bandwidth(matrix)
       # print("solution",solution)
        processed_matrix = np.round(processed_matrix, 5)
        x= np.round(x, 5)
        return jsonify({
            'processed_matrix': processed_matrix.tolist(),
            'is_symmetric': is_symmetric,
            'x':x.tolist(),
            'm_value': m_value
            
        })

    except ValueError as e:
        return jsonify({
            'error': f"Error: {str(e)}"
        })

# Function for matrix processing (e.g., Gaussian elimination)
def process_matrix(matrixx,b):
    start_time = timeit.default_timer()
    n = len(matrixx)
    matrix = np.concatenate((matrixx, np.expand_dims(b, axis=1)), axis=1)
    print(matrix)
    for k in range(n-1):
        for i in range(k + 1, n):
            # Check if the pivot element is zero to avoid division by zero
            if matrix[k][k] == 0:
                raise ValueError("Division par zéro : L'élément pivot est zéro.")

            factor = matrix[i][k] / matrix[k][k]
            for j in range(k, n+1):
                matrix[i][j] -= factor * matrix[k][j]
    end_time = timeit.default_timer()
    execution_time = (end_time - start_time) * 1000
    print(f"Execution time for process_matrix: {execution_time:.6f} seconds")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i, -1]
        for j in range(i + 1, n):
            x[i] -= matrix[i, j] * x[j]
        x[i] /= matrix[i, i]
    return matrix,x




def process_matrixBande(matrixx,b, m):
      start_time = timeit.default_timer()
      
      matrix = np.concatenate((matrixx, np.expand_dims(b, axis=1)), axis=1)
      n = len(matrixx)
      for k in range(n-1):
        for i in range(k + 1, n):
            # Check if the pivot element is zero to avoid division by zero
            if matrix[k][k] == 0:
                raise ValueError("Division par zéro : L'élément pivot est zéro.")

            factor = matrix[i][k] / matrix[k][k]
            

            # Update the loop bounds based on the band matrix
            for j in range(max(0, k - m), min(n-k+1, k + m+2 )+1):
                matrix[i][j] -= factor * matrix[k][j]
      end_time = timeit.default_timer()
      execution_time = (end_time - start_time) * 1000
      print(f"Execution time for process_matrix: {execution_time:.6f} seconds")
      x = np.zeros(n)
      for i in range(n - 1, -1, -1):
        x[i] = matrix[i, -1]
        for j in range(i + 1, n):
            x[i] -= matrix[i, j] * x[j]
        x[i] /= matrix[i, i]
      return matrix,x 


##### gauss pivotage paritel

@app.route('/gaussbivpart')
def gaussbivpart():
    return render_template('gaussAvecPivotagePartiel.html')

@app.route('/eliminategosspp', methods=['POST'])
def eliminate_gauss_pp():
    try:
        n = int(request.form['dimension'])
        matrix = np.array([[float(request.form[f'matrix[{i}][{j}]']) for j in range(n)] for i in range(n)])
        vector = np.array([float(request.form[f'vector[{i}]']) for i in range(n)])
        matrix_type = request.form.get('matrixType')

        if check_non_symmetry(matrix):
            if matrix_type == 'dense':
                result, x = gauss_with_partial_pivoting(matrix, vector)
            elif matrix_type == 'bande':
                m = find_bandwidth(matrix)
                if is_band(np.array(matrix), m):
                    result, x = gauss_with_partial_pivoting_banded(np.array(matrix), m, vector)
                else:
                    raise ValueError("Matrice n'est pas bande.")
        else:
            raise ValueError("La matrice ne doit pas être symétrique.")

        result_html = '<div><div class="row">'

        if matrix_type == 'bande':
            m = find_bandwidth(matrix)
            result_html += f'<div><h3>Valeur de m: {m}</h3></div>'

      

        # Display the processed matrix
        result_html += '<div class="col-12">'
        result_html += '<h3>Résultat: Matrice avec second membre après application de Gauss avec pivotage partiel</h3>'
        result_html += '<table class="table">'
        
        for row in result:
            result_html += '<tr>'
            for elem in row:
                result_html += f'<td>{elem:.5f}</td>'
            result_html += '</tr>'

        result_html += '</table>'
        result_html += '</div>'

        result_html += '</div></div>'

          # Display the solution
        result_html += '<div class="col-12">'
        result_html += '<h3>Solution:</h3>'
        result_html += '<table class="table">'
        result_html += '<tr>'

        for element in x:
            result_html += f'<td>{element:.5f}</td>'

        result_html += '</tr>'
        result_html += '</table>'
        result_html += '</div>'

    except ValueError as e:
        return jsonify({'error': f"Error: {str(e)}"})

    return result_html



def gauss_with_partial_pivoting(A,b):
    rows, cols = A.shape
    A = np.column_stack((A, b))
    for k in range(rows - 1):
        pivot_index = np.argmax(abs(A[k:, k])) + k
        A[[k, pivot_index]] = A[[pivot_index, k]]
        for i in range(k + 1, rows):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
    x = np.zeros(rows)
    for i in range(rows - 1, -1, -1):
        x[i] = A[i, -1]
        for j in range(i + 1, rows):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]
    return A,x

def gauss_with_partial_pivoting_banded(A, m,b):
    rows, cols = A.shape
    A = np.column_stack((A, b))
    for k in range(rows - 1):
        start_index = max(0, k - m)
        end_index = min(rows - 1, k + m)

        pivot_index = np.argmax(abs(A[k, start_index:end_index+1])) + start_index
        A[[k, pivot_index]] = A[[pivot_index, k]]

        for i in range(k + 1, rows):
            factor = A[i, k] / A[k, k]
            A[i, start_index:] -= factor * A[k, start_index:]
    x = np.zeros(rows)
    for i in range(rows - 1, -1, -1):
        x[i] = A[i, -1]
        for j in range(i + 1, rows):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]
    return A,x


########### cholesky 
@app.route('/cholesky')
def homech():
    return render_template('cholesky.html')

@app.route('/decomposechol', methods=['POST'])
def cholesky_decompose():
    try:
        n = int(request.form['matrixSize'])
        matrix = np.array([[float(request.form[f'matrix[{i}][{j}]']) for j in range(n)] for i in range(n)]) 
        vector = np.array([float(request.form[f'vector[{i}]']) for i in range(n)])
        matrix_type = request.form.get('matrixType')
        if not is_positive_definite(matrix):
            raise ValueError("La matrice n'est pas définie positive.")
        # Vérification de la symétrie de la matrice
        if not is_symmetric(matrix):
            raise ValueError("La matrice n'est pas symétrique. La décomposition de Cholesky s'applique uniquement aux matrices symétriques définies positives.")
        m = find_bandwidth(matrix)
        if matrix_type == 'dense':
            lower_cholesky,x = cholesky_decomposition(matrix,vector)
        elif matrix_type == 'bande':
             
            if is_band(np.array(matrix),m):
               
                lower_cholesky,x = cholesky_banded_decomposition(matrix,vector,m)
            else:
                raise ValueError("Matrice n'est pas bande.")
        else:
            raise ValueError("Type de matrice invalide.")
        transposed_matrix_1=lower_cholesky[:, :-1]
        transposed_matrix=np.transpose(transposed_matrix_1)


        result_html = '<div><div class="row">'

        if matrix_type == 'bande':
            m = find_bandwidth(matrix)
            result_html += f'<div><h3>Valeur de m: {m}</h3></div>'

      

        # Display the processed matrix
        result_html += '<div class="col-12">'
        result_html += '<h3>Résultat: Matrice avec second membre après application de Cholesky</h3>'
        result_html += '<table class="table">'
        
        for row in lower_cholesky:
            result_html += '<tr>'
            for elem in row:
                result_html += f'<td>{elem:.5f}</td>'
            result_html += '</tr>'

        result_html += '</table>'
        result_html += '</div>'

        result_html += '</div></div>'

          # Display the solution
        result_html += '<div class="col-12">'
        result_html += '<h3>Solution:</h3>'
        result_html += '<table class="table">'
        result_html += '<tr>'

        for element in x:
            result_html += f'<td>{element:.5f}</td>'

        result_html += '</tr>'
        result_html += '</table>'
        result_html += '</div>'

    except ValueError as e:
        return jsonify({'error': f"Error: {str(e)}"})

    return result_html




    return result_html

def cholesky_decomposition(matrix, b):
    if not is_positive_definite(matrix):
        raise ValueError("La matrice n'est pas définie positive. Assurez-vous que tous les mineurs fondamentaux diagonaux sont positifs.")

    n = len(matrix)
    lower = np.zeros((n, n))  # Ajout d'une colonne pour le vecteur b

    # Concaténer le vecteur b à la matrice
    matrix = np.concatenate((matrix, np.expand_dims(b, axis=1)), axis=1)

    # Calcul de la décomposition de Cholesky
    for j in range(n):
        ljj = matrix[j][j]
        for k in range(j):
            ljj -= lower[j][k] ** 2

        lower[j][j] = np.sqrt(ljj)

        for i in range(j + 1, n):
            lij = matrix[i][j]
            for k in range(j):
                lij -= lower[i][k] * lower[j][k]

            lower[i][j] = lij / lower[j][j]

    # Résolution du système inférieur triangulaire Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = matrix[i, -1]
        for k in range(i):
            y[i] -= lower[i, k] * y[k]
        y[i] /= lower[i, i]

    # Résolution du système supérieur triangulaire L^Tx = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for k in range(i+1, n):
            x[i] -= lower[k, i] * x[k]
        x[i] /= lower[i, i]
    lower = np.concatenate((lower, np.expand_dims(x, axis=1)), axis=1)
    return lower,x


def cholesky_banded_decomposition(matrix, b, m):
    if not is_positive_definite(matrix):
        raise ValueError("La matrice n'est pas définie positive. Assurez-vous que tous les mineurs fondamentaux diagonaux sont positifs.")

    n = len(matrix)
    lower = np.zeros((n, n))

    matrix = np.concatenate((matrix, np.expand_dims(b, axis=1)), axis=1)

    for j in range(n):
        ljj = matrix[j][j]
        for k in range(max(0, j - m), j):
            ljj -= lower[j][k] ** 2

        lower[j][j] = np.sqrt(ljj)

        for i in range(j + 1, min(n, j + m + 1)):
            lij = matrix[i][j]
            for k in range(max(0, j - m), j):
                lij -= lower[i][k] * lower[j][k]

            lower[i][j] = lij / lower[j][j]


    # Résolution du système inférieur triangulaire Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = matrix[i, -1]
        for k in range(i):
            y[i] -= lower[i, k] * y[k]
        y[i] /= lower[i, i]

    # Résolution du système supérieur triangulaire L^Tx = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for k in range(i+1, n):
            x[i] -= lower[k, i] * x[k]
        x[i] /= lower[i, i]
    lower = np.concatenate((lower, np.expand_dims(x, axis=1)), axis=1)
    return lower,x

def is_symmetric(matrix):
    return (matrix == matrix.T).all()


############ Reso Triangulaire 

####### Triangulaire superieure 

def solve_upper_triangular(A, b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = b[i]

        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]

        x[i] /= A[i, i]

    return x


@app.route('/resoTriangSup', methods=['GET', 'POST'])
def resoTriantgSup():
    if request.method == 'POST':
        try:
            size = int(request.form['size'])
            matrix_input = request.form.getlist('matrixElement')
            vector_input = request.form.getlist('vectorBElement')

            # Convert the flattened input to a numpy array
            A = np.array(matrix_input, dtype=float).reshape((size, size))
            b = np.array(vector_input, dtype=float)

            if not is_upper_triangular(A):
                raise ValueError("Donnez une matrice triangulaire supérieure.")

            solution = solve_upper_triangular(A, b)

            # Pass the data to the template
            return render_template('resoTriangSup.html', solution={'A': A.tolist(), 'b': b.tolist(), 'result': solution}, error=None)

        except ValueError as e:
            return render_template('resoTriangSup.html', solution=None, error=f"Error: {str(e)}")

    return render_template('resoTriangSup.html', solution=None, error=None)

########### Reso Triangulaire inferieure 

@app.route('/indextriangInf')
def resoTriangInf():
    return render_template('ResotriangInf.html')

def solve_lower_triangular(A, b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n):
        x[i] = b[i]

        for j in range(i):
            x[i] -= A[i, j] * x[j]

        x[i] /= A[i, i]

    return x

def is_lower_triangular(A):
    rows, cols = A.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            if A[i, j] != 0:
                return False
    return True

@app.route('/indextriangInf', methods=['GET', 'POST'])
def indextriangInfa():
    if request.method == 'POST':
        size = int(request.form['size'])
        matrix_input = request.form.getlist('matrixElement')
        vector_input = request.form.getlist('vectorBElement')

        # Convert the flattened input to a numpy array
        A = np.array(matrix_input, dtype=float).reshape((size, size))
        b = np.array(vector_input, dtype=float)

        if not is_lower_triangular(A):
            return render_template('resotriangInf.html', solution=None, error="Donner une matrice triangulaire inférieure.")

        solution = solve_lower_triangular(A, b)

        # Pass the data to the template
        return render_template('resotriangInf.html', solution={'A': A.tolist(), 'b': b.tolist(), 'result': solution}, error=None)

    return render_template('resotriangInf.html', solution=None, error=None)


###### Reso demi bande inferieure 

@app.route('/resodemiinf')
def resodemiInf():
    return render_template('resodemiinf.html')

def solve_lower_band(matrix, b,m):
    n = len(b)
    x = [0] * n

    for i in range(n):
        x[i] = b[i]

        for j in range(max(0, i - m -1), i):
            x[i] -= matrix[i][j] * x[j]
    if matrix[i][i]!=1:
        x[i] /= matrix[i][i]

    return x


@app.route('/resodemiinf', methods=['GET', 'POST'])
def resodemiinfa():
    if request.method == 'POST':
        try:
            size = int(request.form['size'])

            # Check if matrix and vector inputs are not empty
            matrix_input = []
            for i in range(size):
                row = []
                for j in range(size):
                    element_name = f"matrixElement[{i}][{j}]"
                    if element_name in request.form:
                        row.append(float(request.form[element_name]))
                    else:
                        raise ValueError("Remplir tous les éléments de la matrice.")

                matrix_input.append(row)

            vector_input = [float(request.form[f'vectorBElement[{i}]']) for i in range(size)]

            # Convert the input to a numpy array
            matrix = matrix_input
            b = vector_input
            m = find_bandwidth(matrix)
            is_band = is_lower(matrix, m)

            if not is_band:
                raise ValueError("Donner une matrice demi bande inferieure.")

            solution = solve_lower_band(matrix, b, m)

            # Pass the data to the template
            return render_template('resodemiinf.html', solution={'A': matrix, 'b': b, 'result': solution, 'm': m}, error=None)

        except ValueError as e:
            return render_template('resodemiinf.html', solution=None, error=f"Error: {str(e)}")

    return render_template('resodemiinf.html', solution=None, error=None)

#### resolution demi superieure 

@app.route('/resodemisup')
def resodemisup():
    return render_template('resodemisup.html')

def solve_upper_band(matrix, b, m):
    n = len(b)
    x = [0] * n

    for i in range(n-1,-1,-1):
        x[i] = b[i]

        for j in range(i+1 , min(i + m+1, n)):
            x[i] -= matrix[i][j] * x[j]

    if matrix[i][i]!=1:
        x[i] /= matrix[i][i]

    return x



@app.route('/resodemisup', methods=['GET', 'POST'])
def resodemisupa():
    if request.method == 'POST':
        try:
            size = int(request.form['size'])

            # Check if matrix and vector inputs are not empty
            matrix_input = []
            for i in range(size):
                row = []
                for j in range(size):
                    element_name = f"matrixElement[{i}][{j}]"
                    if element_name in request.form:
                        row.append(float(request.form[element_name]))
                    else:
                        raise ValueError("Remplir tous les éléments de la matrice.")

                matrix_input.append(row)

            vector_input = [float(request.form[f'vectorBElement[{i}]']) for i in range(size)]

            # Convert the input to a numpy array
            matrix = matrix_input
            b = vector_input
            m = find_bandwidth(matrix)
            is_band_test = is_upper(matrix, m)
            if not is_band_test:
                raise ValueError("Donner une matrice demi bande supérieure.")

            solution = solve_upper_band(matrix, b, m)

            # Pass the data to the template
            return render_template('resodemisup.html', solution={'A': matrix, 'b': b, 'result': solution, 'm': m}, error=None)

        except ValueError as e:
            return render_template('resodemisup.html', solution=None, error=f"Error: {str(e)}")

    return render_template('resodemisup.html', solution=None, error=None)


########################## Multiplication matrice matrice 

######### Bande inverse

@app.route('/bandeinverse')
def bandeinverseindex():
    return render_template('bandeinverse.html')

def gauss_jordan_inverse(matrix, m, n):
    for k in range(n):
        pivot = matrix[k][k]

        for j in range(max(0, k - m), min(n - 1, k + m) + 1):
            matrix[k][j] /= pivot

        for i in range(n):
            if i != k:
                factor = matrix[i][k]
                for j in range(max(0, k - m), min(n - 1, k + m) + 1):
                    matrix[i][j] -= factor * matrix[k][j]

    return matrix

def multiplication_bande_par_son_inverse(matrice_bande, m, n):
    resultat = np.zeros((n, n))
    inverse = gauss_jordan_inverse(matrice_bande, m, n)
    for i in range(n):
        for k in range(max(0, i - m), min(n, i + m + 1)):
            resultat[i][i] += matrice_bande[i][k] * inverse[k][i]

    return resultat

def calculate_determinant(matrix):
    return np.linalg.det(matrix)

@app.route('/bandeinverseapp', methods=['GET', 'POST'])
def bandeinverseapp():
    if request.method == 'POST':
        try:
            matrix_size = int(request.form['matrixSize'])
            A_values = [[float(request.form[f'A[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]
            A = np.array(A_values)

            # Check bandwidth
            m = find_bandwidth(A)
            is_band_test = is_band(A_values, m)
            print(is_band_test, m, matrix_size)

            if is_band_test:
                # Check determinant
                determinant = calculate_determinant(A)
                if determinant < 0:
                    return jsonify({'error': 'Le déterminant est négatif. Impossible de poursuivre avec la multiplication.'})

                # Perform multiplication
                result = multiplication_bande_par_son_inverse(A_values, m, matrix_size)
                matrice_inverse = gauss_jordan_inverse(A,m,matrix_size)
                print(matrice_inverse)
                return jsonify({'result': result.tolist(), 'm': m, 'initial_matrix': A.tolist(),'inverse': matrice_inverse.tolist()})
            else:
                return jsonify({'error': 'La matrice ne satisfait pas les conditions de bande requises.'})

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('bandeinverse.html')


######### Bande transpose 

@app.route('/bandetranspose', methods=['GET'])
def bandetransposehome():
    return render_template('bandetranspose.html')

def multiplication_bande_transposer(A, m, n):
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(max(max(0, i - m), max(0, j - m)), min(min(n, m + i + 1), min(n, m + j + 1))):
                result[i, j] += A[i, k] * A[j, k]

    return result

@app.route('/bandetranspose', methods=['POST'])
def bandetranspose():
    try:
        matrix_size = int(request.form['matrixSize'])

        # Retrieve matrix A values
        A_values = [[float(request.form[f'A[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]
        A = np.array(A_values)

        m = find_bandwidth(A)
        is_band_test = is_band(A, m)
        transpose_result = np.transpose(A)
        if is_band_test:
            result = multiplication_bande_transposer(A, m, matrix_size)
            return jsonify({'result': result.tolist(), 'm': m, 'transposeResult': transpose_result.tolist(), 'initialMatrix': A.tolist()})
        else:
            return jsonify({'error': 'La matrice ne satisfait pas les conditions de bande requises.'})

    except Exception as e:
        return jsonify({'error': str(e)})

################# bande x demi bande inf 


@app.route('/bandedemibandinf', methods=['GET'])
def bandedemibandinfindex():
    return render_template('bandedemibandinf.html')

def matriceBande_demiBande(A, B, m1, m2, n):
    m = max(m1, m2)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(0, min(n, m + i + 1)):
            for k in range(j, min(n, m + j + 1)):
                result[i, j] += A[i, k] * B[k, j]
    return result

@app.route('/bandedemibandinf', methods=['POST'])
def bandedemibandinf():
    try:
        matrix_size = int(request.form['matrixSize'])

        # Retrieve matrix A values
        A_values = [[float(request.form[f'A[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]
        A = np.array(A_values)

        # Retrieve matrix B values
        B_values = [[float(request.form[f'B[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]
        B = np.array(B_values)

        # Check if matrices A and B are lower band and upper band matrices, respectively
        m1 = find_bandwidth(A)
        m2 = find_bandwidth(B)
        is_band1 = is_band(A, m1)
        is_lower1 = is_lower(B, m2)
        print(is_lower1,is_band1)
        if is_lower1 and is_band1:
            result = matriceBande_demiBande(A, B, m1, m2, matrix_size)
            return jsonify({'result': result.tolist(),'m1':m1,'m2':m2})
        else:
            return jsonify({'error': 'La matrice ne satisfait pas les conditions de bande requises.'})

    except Exception as e:
        return jsonify({'error': str(e)})

############ demi bande inf x demi bande sup 

@app.route('/bandinfsup', methods=['GET'])
def bandinfsupindex():
    return render_template('bandinfsup.html')

def multiplication_demiBandeInf_demiBandeSup(A, B, m1, m2, n):
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - m1), min(n, m2 + i + 1)):
            for k in range(max(max(0, i - m1), max(0, j - m2)), min(i + 1, j + 1)):
                result[i, j] += A[i, k] * B[k, j]
    return result

@app.route('/bandinfsup', methods=['GET', 'POST'])
def bandinfsup():
    if request.method == 'POST':
        try:
            matrix_size = int(request.form['matrixSize'])

            # Retrieve matrix A values
            A_values = [[float(request.form[f'A[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]
            A = np.array(A_values)

            # Retrieve matrix B values
            B_values = [[float(request.form[f'B[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]
            B = np.array(B_values)

            # Check if matrices A and B are lower and upper band matrices, respectively
            m1 = find_bandwidth(A)
            m2 = find_bandwidth(B)
            is_lower_test = is_lower(A, m1)
            is_upper_test = is_upper(B, m2)

            if is_lower_test and is_upper_test:
                result = multiplication_demiBandeInf_demiBandeSup(A, B, m1, m2, matrix_size)
                return jsonify({'result': result.tolist(), 'm1': m1, 'm2': m2})
            else:
                return jsonify({'error': 'La matrice ne satisfait pas les conditions de bande requises.'})

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('bandinfsup.html')


####################### matrice vecteur 

############# dense vecteur


@app.route('/densevecteur', methods=['GET'])
def indexdensevecteur():
        return render_template('densevecteur.html')

def matrix_vector_multiplication(matrix, vector):
    if matrix.shape[1] != len(vector):
        raise ValueError("Les dimensions de la matrice et du vecteur sont incompatibles pour effectuer la multiplication.")

    result = []

    for i in range(len(matrix)):
        row_result = sum(a * b for a, b in zip(matrix[i], vector))
        result.append(row_result)

    return result

@app.route('/denseVecteur', methods=['GET', 'POST'])
def matrix_vector_multiply():
    
    try:
        matrix_size = int(request.form['matrixSize'])
        vector_size = int(request.form['matrixSize'])

        # Récupérer les valeurs de la matrice
        matrix_values = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
             for j in range(matrix_size):  # Adjusted loop condition here
                key = f'matrix[{i}][{j}]'
                matrix_values[i, j] = float(request.form[key])

        # Retrieve vector values
        vector_values = [float(request.form[f'vector[{i}]']) for i in range(vector_size)]

        # Create numpy arrays
        matrix = np.array(matrix_values)
        vector = np.array(vector_values)

        # Effectuer la multiplication
        result = matrix_vector_multiplication(matrix, vector)

        return render_template('DenseVecteur.html', result=result)

    except ValueError as e:
        error = str(e)
        return render_template('DenseVecteur.html', error=error)

######### triang inf vecteur

@app.route('/infvecteur', methods=['GET'])
def infvecteurindex():
        return render_template('infvecteur.html')

def matrix_inf_vector_multiplication(matrix, vector):
    n = len(vector)
    if n != len(matrix):
        raise ValueError("Les dimensions de la matrice et du vecteur sont incompatibles pour effectuer la multiplication.")
    result = np.zeros(n)

    for i in range(n):
        for j in range(i + 1):
            result[i] += matrix[i, j] * vector[j]

    return result

@app.route('/infvecteurapp', methods=['POST'])
def infvecteurapp(): 
    try:
        matrix_size = int(request.form['matrixSize'])
        vector_size = int(request.form['matrixSize'])

        # Récupérer les valeurs de la matrice
        matrix_values = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
             for j in range(matrix_size):  # Adjusted loop condition here
                key = f'matrix[{i}][{j}]'
                matrix_values[i, j] = float(request.form[key])

        # Retrieve vector values
        vector_values = [float(request.form[f'vector[{i}]']) for i in range(vector_size)]

        matrix = np.array(matrix_values)
        vector = np.array(vector_values)
        # Perform multiplication
        if not is_lower_triangular(matrix):
            return render_template('infvecteur.html', solution=None, error="Donner une matrice triangulaire inférieure.")
        result = matrix_inf_vector_multiplication(matrix, vector)

        return render_template('infvecteur.html', result=result)
    except ValueError as e:
        error = str(e)
        return render_template('infvecteur.html', error=error)

########### traing sup 

@app.route('/supvecteur', methods=['GET'])
def supvecteurindex():
    return render_template('supvecteur.html')

def matrix_sup_vector_multiplication(matrice, vecteur):
    if len(matrice) != len(vecteur):
        raise ValueError("Les dimensions de la matrice et du vecteur sont incompatibles pour effectuer la multiplication.")

    dimension = len(vecteur)
    resultat = [0] * dimension

    for i in range(dimension):
        for j in range(i, dimension):
            resultat[i] += matrice[i][j] * vecteur[j]

    return resultat

@app.route('/supvecteurapp', methods=['POST'])
def supvecteurapp(): 
    try:
        matrix_size = int(request.form['matrixSize'])
        vector_size = int(request.form['matrixSize'])

        # Récupérer les valeurs de la matrice
        matrix_values = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
             for j in range(matrix_size):  # Adjusted loop condition here
                key = f'matrix[{i}][{j}]'
                matrix_values[i, j] = float(request.form[key])

        # Retrieve vector values
        vector_values = [float(request.form[f'vector[{i}]']) for i in range(vector_size)]
        matrix = np.array(matrix_values)
        vector = np.array(vector_values)
        if not is_upper_triangular(matrix):
             return render_template('supvecteur.html', solution=None, error="Donner une matrice triangulaire supérieure.")
        # Perform multiplication
        result = matrix_sup_vector_multiplication(matrix, vector)

        return render_template('supvecteur.html', result=result)
    except ValueError as e:
        error = str(e)
        return render_template('supvecteur.html', error=error)
    
################# demi bande sup x vecteur

@app.route('/demisupvecteur', methods=['GET'])
def demisupvecteurindex():
    return render_template('demisupvecteur.html')    

def multiply_upper_band_matrix_vector(matrix, vector, m):
    n = len(matrix)
    result = [0] * n
    for i in range(n):
        result[i] = 0
        for j in range(i, min(i + m + 1, n)):
            result[i] += matrix[i][j] * vector[j]
    return result

@app.route('/demisupvecteurapp', methods=['POST'])
def demisupvecteurapp():
    try:
        matrix_size = int(request.form['Size'])
        vector_size = int(request.form['Size'])

        # Retrieve matrix values
        matrix_values = [[float(request.form[f'matrix[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]

        # Retrieve vector values
        vector_values = [float(request.form[f'vector[{i}]']) for i in range(vector_size)]

        # Create numpy arrays
        matrix = np.array(matrix_values)
        vector = np.array(vector_values)

        # Check if the matrix is upper band
        m = find_bandwidth(matrix)
        is_band = is_upper(matrix, m)  # Pass 'm' directly for now, you may need to adjust this

        if is_band:
            result = multiply_upper_band_matrix_vector(matrix, vector, m)
            # Render the 'result.html' template with the calculated result
            return jsonify({'result': result,'m':m})  # Retourne le résultat dans un format JSON
        else:
            return jsonify({'error': "La matrice n'est pas une matrice bande supérieure."})

    except Exception as e:
        return jsonify({'error': str(e)})    

#################### demi inf x vecteur 

@app.route('/demiinfvecteur', methods=['GET'])
def demiinfvecteurindex():
    return render_template('demiinfvecteur.html')   

def multiply_lower_band_matrix_vector(matrix, vector, m):
    n=len(matrix)
    result = [0] * n
    for i in range(n):
        result[i] = 0
        for j in range(max(0, i-1),i+1):
            result[i] += matrix[i][j] * vector[j]
    return result

@app.route('/demiinfvecteurapp', methods=['POST'])
def demiinfvecteurapp():
    try:
        matrix_size = int(request.form['Size'])
        vector_size = int(request.form['Size'])

        # Retrieve matrix values
        matrix_values = [[float(request.form[f'matrix[{i}][{j}]']) for j in range(matrix_size)] for i in range(matrix_size)]

        # Retrieve vector values
        vector_values = [float(request.form[f'vector[{i}]']) for i in range(vector_size)]

        # Create numpy arrays
        matrix = np.array(matrix_values)
        vector = np.array(vector_values)

       
        m = find_bandwidth(matrix)
        is_band = is_lower(matrix,m)
        
        if is_band :
            result = multiply_lower_band_matrix_vector(matrix, vector, m)
            return jsonify({'result': result,'m':m})
        else:  
            return jsonify({'error': "La matrice n'est pas une matrice bande inférieure."})

    except Exception as e:
        return jsonify({'error': str(e)})
    

################### iteratives 

######### Gauss Seidel 

def calculate_G(matrix):
    # Calculate D, L, and U matrices for Gauss-Seidel
    D_matrix = np.diag(np.diag(matrix))
    L_matrix = -1*np.tril(matrix, k=-1)  
    U_matrix = -1*np.triu(matrix, k=1)
    # Calculate G matrix for Gauss-Seidel (corrected formula)
    inverse_DL = np.linalg.inv(D_matrix - L_matrix)
    G_matrix = np.dot(inverse_DL, U_matrix)

    return G_matrix

def eigenvalues_of_G(matrix_G, threshold=1e-4):
    # Calculate eigenvalues of G
    eigenvalues_G, _ = np.linalg.eig(matrix_G)
    
    # Set eigenvalues close to zero to exactly zero
    eigenvalues_G[np.abs(eigenvalues_G) < threshold] = 0

    return eigenvalues_G

def is_diagonally_dominant(matrix):
    rows, cols = matrix.shape

    for i in range(rows):
        row_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])

        if np.abs(matrix[i, i]) < row_sum:
            return False

    return True

def calculate_determinant(matrix):
    return np.linalg.det(matrix)

def gauss_seidel_method_epsilon(A, b, epsilon=1e-6):
    n = len(b)
    x = np.zeros(n)
    max_difference = epsilon + 1
    while max_difference > epsilon:
        y = np.copy(x)
        max_difference = 0  # Reset max_difference for each iteration
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            s = (b[i] - s) / A[i, i]

            if np.abs(s - y[i]) > max_difference:
                max_difference = np.abs(s - y[i])

            x[i] = s

    return x

def gauss_seidel_method_iteration(A, b, iteration):
    n = len(b)
    x = np.zeros(n)

    for _ in range(iteration):
        y = np.copy(x)

        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            s = (b[i] - s) / A[i, i]

            x[i] = s

    return x

@app.route('/gs')
def indexgs():
    return render_template('gaussseidel.html')


@app.route('/solvegs', methods=['POST'])
def solvegs():
    try:
        # Retrieve matrix A and vector b from the form
        matrix_size = int(request.form['matrixSize'])
        vector_size = int(request.form['vectorSize'])
        iteration = int(request.form.get('maxIterations', 0))
        iteration_test = request.form.get('iterationsCheckbox')
        print(iteration, iteration_test)
        A = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                A[i, j] = float(request.form[f'matrix[{i+1}][{j+1}]'])

        b = np.zeros(vector_size)
        for i in range(vector_size):
            b[i] = float(request.form[f'vector[{i+1}]'])
        print(A,b)
        for i in range(1, vector_size + 1):
            b[i-1] = float(request.form[f'vector[{i}]'])
            # Check if the determinant of matrix A is non-zero
        determinant = calculate_determinant(A)
        if determinant <= 0:
            return jsonify({'error': 'Le déterminant de la matrice est nul. Impossible de procéder avec la méthode de Gauss-Seidel.'})

        # Check if matrix A is diagonally dominant
        if not is_diagonally_dominant(A):
            # Calculate G matrix for Gauss-Seidel
            matrix_G = calculate_G(A)

          # Calculate eigenvalues of G with a threshold
            eigenvalues_G = eigenvalues_of_G(matrix_G)
           # Check if the maximum eigenvalue is strictly less than 1
            max_eigenvalue_G = np.max(np.abs(eigenvalues_G))
            if max_eigenvalue_G < 1 :  
               if iteration_test:
                  msg = f"La matrice n'est pas dominante ρ(G) = {max_eigenvalue_G:.1f} < 1. La méthode de Gauss-Seidel converge."
                  result = gauss_seidel_method_iteration(A, b, iteration)
                  return jsonify({'matrix_G': matrix_G.tolist(),"msg":msg, 'result': result.tolist()})
               else:
                   msg = f"La matrice n'est pas dominante ρ(G) = {max_eigenvalue_G:.1f} < 1. La méthode de Gauss-Seidel converge."
                   result = gauss_seidel_method_epsilon(A, b)
                   return jsonify({'matrix_G': matrix_G.tolist(),"msg":msg, 'result': result.tolist()})
            else:
               msg =f"La matrice n'est pas dominante ρ(G) = {max_eigenvalue_G:.1f} > 1. La méthode de Gauss-Seidel diverge."
               return jsonify({"msg":msg,'result': None})
        else: 
            if iteration_test:
                  msg = "Matrice dominante, la méthode de Gauss-Seidel converge."
                  result = gauss_seidel_method_iteration(A, b, iteration)
                  
            else : 
                   msg = "Matrice dominante, la méthode de Gauss-Seidel converge."
                   result = gauss_seidel_method_epsilon(A, b)
        return jsonify({"msg":msg, 'result': result.tolist()})

            
    except Exception as e:
        return jsonify({'error': str(e)})

################### Jabobi

def calculate_J(matrix):
    # Calculate M and N matrices
    diagonal_matrix = np.diag(np.diag(matrix))
    off_diagonal_matrix = matrix - diagonal_matrix

    # Calculate D matrix
    D_matrix = diagonal_matrix

    # Calculate E and F matrices
    E_matrix = np.tril(off_diagonal_matrix, k=-1)
    F_matrix = np.triu(off_diagonal_matrix, k=1)

    # Calculate J matrix
    inverse_D = np.linalg.inv(D_matrix)
    J_matrix = -np.dot(inverse_D, E_matrix + F_matrix)

    return J_matrix

def eigenvalues_of_J(matrix_J, threshold=1e-4):
    # Calculate eigenvalues of J
    eigenvalues_J, _ = np.linalg.eig(matrix_J)
    
    # Set eigenvalues close to zero to exactly zero
    eigenvalues_J[np.abs(eigenvalues_J) < threshold] = 0

    return eigenvalues_J

def is_diagonally_dominant(matrix):
    rows, cols = matrix.shape

    for i in range(rows):
        row_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])

        if np.abs(matrix[i, i]) < row_sum:
            return False

    return True

def calculate_determinant(matrix):
    return np.linalg.det(matrix)

def jacobi_method_epsilon(A, b,epsilon=1e-6):
    n = len(b)
    x = np.zeros(n)  # Initial guess for solution

    max_difference = epsilon + 1
    while max_difference > epsilon:
        y = np.copy(x)
        max_difference = 0  # Reset max_difference for each iteration

        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            y[i] = (b[i] - s) / A[i, i]

        max_difference=np.max(np.abs(y - x))
         

        x = np.copy(y)

    return x

def jacobi_method_iteration(A, b,iteration):
    n = len(b)
    x = np.zeros(n)  

    
    for i in range(iteration):
        y = np.copy(x)

        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            y[i] = (b[i] - s) / A[i, i]

        x = np.copy(y)

    return x




@app.route('/jacobi')
def indexjacobi():
    return render_template('jacobi.html')


@app.route('/solvejacobi', methods=['POST'])
def solvejacobi():
    try:
        matrix_size = int(request.form['matrixSize'])
        vector_size = int(request.form['vectorSize'])
        iteration = int(request.form.get('maxIterations', 0))
        iteration_test = request.form.get('iterationsCheckbox')

        A = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                A[i, j] = float(request.form[f'matrix[{i+1}][{j+1}]'])

        b = np.zeros(vector_size)
        for i in range(vector_size):
            b[i] = float(request.form[f'vector[{i+1}]'])
        print(A,b)
        for i in range(1, vector_size + 1):
            b[i-1] = float(request.form[f'vector[{i}]'])

        determinant = calculate_determinant(A)
        if determinant <= 0:
            return jsonify({'error': 'Le déterminant de la matrice est nul. Impossible de procéder avec la méthode de Jacobi.'})

        if not is_diagonally_dominant(A):
           
            matrix_J = calculate_J(A)
            eigenvalues_J = eigenvalues_of_J(matrix_J)
            max_eigenvalue_J = np.max(np.abs(eigenvalues_J))

            if max_eigenvalue_J < 1:
                if iteration_test:
                    msg = f"La matrice n'est pas dominante ρ(J) = {max_eigenvalue_J:.1f} < 1. La méthode de Jacobi converge."
                    result = jacobi_method_iteration(A, b, iteration)
                    return jsonify({'matrix_J': matrix_J.tolist(), "msg": msg, 'result': result.tolist()})
                else:
                    msg = f"La matrice n'est pas dominante ρ(J) = {max_eigenvalue_J:.1f} < 1. La méthode de Jacobi converge."
                    result = jacobi_method_epsilon(A, b)
                    return jsonify({'matrix_J': matrix_J.tolist(), "msg": msg, 'result': result.tolist()})
            else:
                msg = f"La matrice n'est pas dominante ρ(J) = {max_eigenvalue_J:.1f} > 1. La méthode de Jacobi diverge."
                return jsonify({"msg": msg, 'result': None})
        else:
            if iteration_test:
                msg = "Matrice dominante. La méthode de Jacobi converges."
                result = jacobi_method_iteration(A, b, iteration)
            else:
                msg = "Matrice dominante. La méthode de Jacobi converges."
                result = jacobi_method_epsilon(A, b)

            return jsonify({"msg": msg, 'result': result.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

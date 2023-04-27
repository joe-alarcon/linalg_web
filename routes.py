import json
import re
from types import SimpleNamespace
from app import app, db, login_manager
from flask import request, render_template, flash, redirect, url_for, jsonify, session
from flask_login import current_user, login_user, logout_user, login_required, UserMixin
from forms import *
from werkzeug.urls import url_parse
from models import *
from matrix import *

matrix_array = []

"""
m8.run_all()
matrix_array.append(m8)
example_matrices()
for i in range(1, 21):
    matrix_array.append(eval(f"m{i}"))
"""


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':
        return redirect('/')

    return render_template('home.html')


@app.route('/make_new', methods=['GET', 'POST'])
def make_new():
    return render_template('make_new_matrix.html', m_list=matrix_array, empty=len(matrix_array)==0)


@app.route('/receive', methods=['GET', 'POST'])
def receive():
    if request.method == 'POST':
        data = request.get_json()
        parsed_data = jsonify(data).get_data()
        session['2d_array'] = parsed_data
        return redirect(url_for('create_m'))
    return redirect(url_for('home'))


@app.route('/create', methods=['GET', 'POST'])
def create_m():
    array = json.loads(session['2d_array'])
    aug = array.pop()
    try:
        for i in range(len(array)):
            for j in range(len(array[0])):
                array[i][j] = get_value(array[i][j])
    except ValueError:
        flash('Invalid inputs')
        return redirect(url_for('make_new'))
    m = Matrix('m', 'create', array, True, aug)
    m.run_all()
    curr_size = len(matrix_array)
    matrix_array.append(m)
    return redirect(f'/matrix{curr_size}')


@app.route('/matrix<int:num>', methods=['GET', 'POST'])
def matrix(num):
    try:
        m = matrix_array[num]
        return render_template('matrix.html', matrix=m, clean=clean_number, m_list=matrix_array, empty=len(matrix_array)==0)
    except IndexError:
        flash(f'No matrix with id {num} in memory.')
        return redirect(url_for('make_new'))


def check_which_matrix(string):
    if bool(re.search(r"m\d", string)):
        start_index = string.index('m') + 1
        end_index = start_index + 1
        while end_index < len(string) and bool(re.fullmatch(r"\d", string[end_index])):
            end_index += 1
        numbers = string[start_index:end_index]
        try:
            num = eval(numbers)
            assert isinstance(num, int)
        except AssertionError:
            raise ValueError
        return num


@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    input_form = CalcInForm()

    if input_form.validate_on_submit():
        list_separated_input_string = input_form.text_in_field.data.split(" ")

        for i in range(len(list_separated_input_string)):
            s = list_separated_input_string[i]
            if disallowed_terms(s):
                flash('Invalid inputs: disallowed.')
                return redirect(url_for('calculator'))

            num = check_which_matrix(s)
            if isinstance(num, int):
                try:
                    dot_index = s.index('.')
                    if isinstance(dot_index, int):
                        list_separated_input_string[i] = f"matrix_array[{num}]" + s[dot_index:len(s)]
                except ValueError:
                    list_separated_input_string[i] = f"matrix_array[{num}]"

        parsed_string = " ".join(list_separated_input_string)
        try:
            result = eval(parsed_string)
        #exec(parsed_string, {"transpose": Matrix.transpose, "inverse": Matrix.invert, "matrix_array": matrix_array})
        except SyntaxError or NameError:
            flash("Invalid Inputs: error.")
            return redirect(url_for('calculator'))

        if result is not None:
            curr_size = len(matrix_array)
            matrix_array.append(result)
            return redirect(f'/matrix{curr_size}')
        return redirect(url_for('calculator'))

    return render_template('calculator.html', matrices=matrix_array, in_form=input_form, m_list=matrix_array, empty=len(matrix_array)==0)


@app.route('/clear',  methods=['GET', 'POST'])
def clear():
    matrix_array.clear()
    flash("Memory has been cleared")
    return redirect(url_for('home'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have logged out.')
    return redirect('/')


@login_manager.user_loader
def load_user(u_id):
    return User.query.get(int(u_id))


@login_manager.unauthorized_handler
def unauthorized():
    flash('Unauthorized')
    return redirect("/")


"""
@app.route('/receive', methods=['GET', 'POST'])
def receive():
    if request.method == 'POST':
        data = request.get_json()
        parsed_data = jsonify(data).get_data()
        array = json.loads(parsed_data)
        aug = array.pop()
        try:
            for i in range(len(array)):
                for j in range(len(array[0])):
                    array[i][j] = get_value(array[i][j])
        except ValueError:
            flash('Invalid inputs')
            return redirect('/')
        m = Matrix('m', 'create', array, True, aug)
        session['matrix_array'].append(m)
        index = len(session['matrix_array']) - 1
        m.run_all()
        print("Before", session['matrix_array'])
        return redirect(f'/matrix{index}')


@app.route('/matrix<int:index>', methods=['GET', 'POST'])
def display_matrix(index):
    print("After", session['matrix_array'])
    m = session['matrix_array'][index]
    return render_template('matrix.html', matrix=m)

"""



"""
@app.route('/receive', methods=['GET', 'POST'])
def receive():
    if request.method == 'POST':
        data = request.get_json()
        parsed_data = jsonify(data).get_data()
        session['2d_array'] = parsed_data
        return redirect(url_for('create_m'))


@app.route('/matrix', methods=['GET', 'POST'])
def create_m():
    array = json.loads(session['2d_array'])
    aug = array.pop()
    try:
        for i in range(len(array)):
            for j in range(len(array[0])):
                array[i][j] = get_value(array[i][j])
    except ValueError:
        flash('Invalid inputs')
        return redirect('/')
    m = Matrix('m', 'create', array, True, aug)
    m.run_all()
    return render_template('matrix.html', matrix=m)

"""
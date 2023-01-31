from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, BooleanField, RadioField, PasswordField
from wtforms.validators import DataRequired, Email, EqualTo


class CalcInForm(FlaskForm):
    text_in_field = StringField('What would you like to calculate?', validators=[DataRequired()])
    submitButton = SubmitField('Submit')

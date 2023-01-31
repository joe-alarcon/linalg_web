const dropEl = document.getElementById("dropdown-content-list")
const calcEl = document.getElementById("dropdown-content-list-calculator")
dropDownMenuItems = JSON.parse(localStorage.getItem("dropDownItems"))

if (false) { //dropDownMenuItems
    renderDropDown(calcEl)
} else {
    dropDownMenuItems = ''
}

function renderDropDown(elIsPresent) {
    let rendererFunction = renderWithArray(dropDownMenuItems)
    if (false) {
        calcEl.innerHTML = rendererFunction('table')
    }
    dropEl.innerHTML = rendererFunction('list')
}

function renderWithArray(array) {
    function renderAs(element) {

        let toReturn = ''

        if (element == 'list') {
            for (i = 0; i < array.length; i++) {
                toReturn += `<li> ${array[i]} </li>`
            }
        } else if (element == 'table') {
            toReturn += "<tr>"
            for (i = 0; i < array.length; i++) {
                toReturn += `<th> ${array[i]} </th>`
            }
            toReturn += "</tr>"
            toReturn += "<tr>"
            for (i = 0; i < array.length; i++) {
                 toReturn += `<td> {{ rendering(matrices[${i}]) }} </td>`
            }
            toReturn += "</tr>"
        }
        return toReturn
    }
    return renderAs
}


const clearBTN = document.getElementById("clear-btn")
clearBTN.addEventListener("dblclick", function() {
    clearLocalStorage()
})

function clearLocalStorage() {
    localStorage.removeItem('dropDownItems')
    localStorage.removeItem('dropDownCount')
    fetch("http://127.0.0.1:5000/clear",
                {
                    method: 'POST',
                    headers: {
                        'Content-type': 'application/json',
                        'Accept': 'application/json'
                    }
                }).then(response => {
                        if (response.ok && response.redirected) {
                             window.location = response.url
                        } else {
                             alert("Error")
                        }
                    }
                    ).catch((err) => console.error(err))
}



//Deprecated code from html files


//<!-- {{ "\( \\newenvironment{amatrix}[1]{% \n \left(\\begin{array}{@{}*{#1}{c}|c@{}} }{% \n \end{array}\\right)} \)" }} -->
//<!-- \[ \begin{amatrix}{{ '{' + (matrix.w - 1).__str__() + '}' }} {{ matrix.reduced_matrix.__repr__() }} \end{amatrix} \] -->

/* <!-- <table>
        {% if matrix.augm %}
        <colgroup>
            <col span="{{ matrix.w - 1 }}" class="matrix">
            <col class="augm">
        </colgroup>
        {% endif %}
        {% for row in matrix.get_values_rows(matrix.h, matrix.w): %}
        <tr>
            {% for item in row: %}
                <td> {{ item }} </td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table> --> */

const rowIn = document.getElementById("rows")
const colIn = document.getElementById("cols")
const augIn = document.getElementById("aug")
const tableEl = document.getElementById("table")
//const flashEl = document.getElementById("flash")

let dropDownMenuItems = JSON.parse(localStorage.getItem("dropDownItems"))
if (!dropDownMenuItems) {
    dropDownMenuItems = []
}

let renderCount = JSON.parse(localStorage.getItem("dropDownCount"))
if (!renderCount) {
    renderCount = 0
}

let rows
let cols
let aug

//------------------------
//Functions for getting size input and generating input table
const sizeBTN = document.getElementById("size-btn")
sizeBTN.addEventListener("click", function() {
    rows = parseInt(rowIn.value)
    cols = parseInt(colIn.value)
    aug = augIn.checked
    rowIn.value = ""
    colIn.value = ""
    augIn.checked = false
    //flashEl.innerHTML += `<li> r = ${rows}, c = ${cols}, aug = ${aug} </li>`
    tableEl.innerHTML = parseSize(rows, cols, aug)
})

function parseSize(r, c, a) {
    completeSTR = ""
    if (a) {
        c += 1
        completeSTR += `<colgroup>
            <col span="${c-1}" style="background-color:white">
            <col style="background-color:grey">
        </colgroup>`
    }

    for (i = 0; i < r; i += 1) {
        rowHTML = "<tr>"
        for (j = 0; j < c; j += 1) {
            rowHTML += `<th><input type="text" id="${i},${j}"></th>`
        }
        rowHTML += "</tr>"
        completeSTR += rowHTML
    }
    return completeSTR
}
//------------------------------


//Functions for reading and parsing user input from table
const pyEl = document.getElementById("submit-m")
pyEl.addEventListener("click", function() {
    let theArray = get2DArray(rows, cols, aug)
    if (checkFull(theArray.slice(0, -1))) {
        passToPython(theArray, rows, cols, aug)
        updateDropDown()
        console.log("accepted")
    } else {
        alert("No input detected in some fields - input is required")
    }
})

function get2DArray(r, c, a) {
    if (a) {
      c += 1
    }
    let m = []
        for (i = 0; i < r; i += 1) {
        let row = []
            for (j = 0; j < c; j += 1) {
                let id = i + "," + j
                let inputField = document.getElementById(id).value
                row.push(inputField)
            }
        m.push(row)
    }
    m.push(a)
    return m
}

function checkFull(array) {
    if (array.length === 0) {
        return false
    }
    for (i = 0; i < array.length; i += 1) {
        let curr = array[i]
        for (j = 0; j < curr.length; j += 1) {
            if (!curr[j]) {
                return false
            }
        }
    }
    return true
}

function passToPython(array) {
    fetch("http://127.0.0.1:5000/receive",
            {
                method: 'POST',
                headers: {
                    'Content-type': 'application/json',
                    'Accept': 'application/json'
                },
            // Stringify the payload into JSON:
            body: JSON.stringify(array)}).then(response => {
                    if (response.ok && response.redirected) {
                         window.location = response.url
                    } else {
                         alert("Invalid Input")
                    }
                }
                ).catch((err) => console.error(err))
}

//Promise: after fetch is completed, it returns a promise - code flow with .then() methods
//response, in this case, is the fulfilled execution of the promise

//Update and Render DropDown
function updateDropDown() {
    if (!dropDownMenuItems) {
        dropDownMenuItems = []
    }
    dropDownMenuItems.push( `<a href="/matrix${renderCount}"> Matrix ${renderCount} </a>` )
    renderCount += 1
    localStorage.setItem("dropDownItems", JSON.stringify(dropDownMenuItems))
    localStorage.setItem("dropDownCount", JSON.stringify(renderCount))
}

//----------------------------------------


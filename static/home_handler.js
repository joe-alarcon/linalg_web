/* const listEl = document.getElementById("dropdown-content-list-body")
dropDownMenuItems = JSON.parse(localStorage.getItem("dropDownItems"))

if (dropDownMenuItems) {
    renderDropDown()
} else {
    dropDownMenuItems = ''
}

function renderDropDown() {
    listEl.innerHTML = dropDownMenuItems
} */

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

document.getElementById("submitBtn").addEventListener("click", () => {
    const formData = {
        year: document.getElementById("year").value,
        yearBand: document.getElementById("yearBand").value,
        usePeak: document.getElementById("usePeak").value,
        ageBallpark: document.getElementById("ageBallpark").value,
        sex: document.getElementById("sex").value,
        genderCatMasc: document.getElementById("genderCatMasc").checked,
        genderCatNeutMasc: document.getElementById("genderCatNeutMasc").checked,
        genderCatNeut: document.getElementById("genderCatNeut").checked,
        genderCatNeutFem: document.getElementById("genderCatNeutFem").checked,
        genderCatFem: document.getElementById("genderCatFem").checked,
        neverTop: document.getElementById("neverTop").value,
        numLo: document.getElementById("numLo").value,
        numHi: document.getElementById("numHi").value,
        numResults: document.getElementById("numResults").value,
    };

    fetch("/peak", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => populateResultsTable(data))
    .catch(error => console.error('Error:', error));
});

function populateResultsTable(data) {
    const tableHeader = document.getElementById("resultsTableHeader");
    const tableBody = document.getElementById("resultsTableBody");

    // Clear existing table data
    tableHeader.innerHTML = "";
    tableBody.innerHTML = "";

    // Create table headers
    const headers = Object.keys(data[0]);
    headers.forEach(header => {
        const th = document.createElement("th");
        th.textContent = header;
        tableHeader.appendChild(th);
    });

    // Create table rows
    data.forEach(row => {
        const tr = document.createElement("tr");
        headers.forEach(header => {
            const td = document.createElement("td");
            td.textContent = row[header];
            tr.appendChild(td);
        });
        tableBody.appendChild(tr);
    });
}

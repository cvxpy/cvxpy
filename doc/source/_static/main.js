$(document).ready(function() {
  function createRadioButtons() {
    var radioButtons = `
      <h3>Curvature Filter</h3>
      <form id="filterForm">
        <label>
          <input type="radio" name="filter" value="all" checked> All
        </label>
        <label>
          <input type="radio" name="filter" value="convex"> Convex
        </label>
        <label>
          <input type="radio" name="filter" value="concave"> Concave
        </label>
      </form>
    `;
    var dropdown = document.querySelector('.functions-filter .sd-card-text');

    if (dropdown) {
      dropdown.innerHTML = radioButtons;
    }

  }

  function initializeMainTable() {
    var table = $('table.atomic-functions').DataTable();
    $('input[name="filter"]').change(function() {
      var selectedValue = $(this).val();
      if (selectedValue === "all") {
        table.search('').columns().search('').draw();
      } else {
        table.column(4).search(selectedValue).draw();
      }
    });
  }

function createDCPTable(tableClass, typeValue) {
  var newTable = $(`table.${tableClass}`);
  const curvatures = ['convex', 'concave', 'affine'];
  if (typeValue === "elementwise") {
    curvatures.push('constant');
  }

  const dcpData = $('table.atomic-functions').DataTable().data().toArray().filter(row => {
    const curvatureValue = $(row[4]).text().trim();
    const rowTypeValue = $(row[6]).text().trim();
    return curvatures.includes(curvatureValue) && rowTypeValue === typeValue;
  });

  newTable.DataTable({
    data: dcpData,
    columns: [
      { title: "Function" },
      { title: "Meaning" },
      { title: "Domain" },
      { title: "Sign" },
      { title: "Curvature" },
      { title: "Monotonicity" },
    ]
  });
}

function createMatrixDCPTable(tableClass, typeValue) {
  var newTable = $(`table.${tableClass}`);
  const curvatures = ['affine'];

  const dcpData = $('table.atomic-functions').DataTable().data().toArray().filter(row => {
    const curvatureValue = $(row[4]).text().trim();
    const rowTypeValue = $(row[6]).text().trim();
    return curvatures.includes(curvatureValue) && rowTypeValue === typeValue;
  });

  newTable.DataTable({
    data: dcpData,
    columns: [
      { title: "Function" },
      { title: "Meaning" },
      { title: "Domain" },
      { title: "Sign", visible: false },
      { title: "Curvature" },
      { title: "Monotonicity" },
    ]
  });
}

function createDGPTable(tableClass, typeValue) {
  var newTable = $(`table.${tableClass}`);
  const curvatures = ['log-log affine', 'log-log convex', 'log-log concave', 'constant'];

  const dcpData = $('table.atomic-functions').DataTable().data().toArray().filter(row => {
    const curvatureValue = $(row[4]).text().trim();
    const rowTypeValue = $(row[6]).text().trim();
    return curvatures.includes(curvatureValue) && rowTypeValue === typeValue;
  });

  newTable.DataTable({
    data: dcpData,
    columns: [
      { title: "Function" },
      { title: "Meaning" },
      { title: "Domain" },
      { title: "Sign", visible: false },
      { title: "Curvature" },
      { title: "Monotonicity" },
    ]
  });
}

  createRadioButtons();
  initializeMainTable();
  createDCPTable("scalar-dcp", "scalar");
  createDCPTable("element-dcp", "elementwise");
  createMatrixDCPTable("matrix-dcp", "matrix");
  createDGPTable("scalar-dgp", "scalar");
  createDGPTable("element-dgp", "elementwise");
  createDGPTable("matrix-dgp", "matrix");

});

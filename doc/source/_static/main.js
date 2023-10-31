$(document).ready(function() {
  function createRadioButtons() {
    var radioButtons = `
      <h5>Curvature</h5>
      <form id="curvatureFilter">
        <div></div>
        <label>
          <input type="radio" name="curvature" value="all" checked> All
        </label>
        <label>
          <input type="radio" name="curvature" value="convex"> Convex
        </label>
        <label>
          <input type="radio" name="curvature" value="concave"> Concave
        </label>
        <label>
          <input type="radio" name="curvature" value="affine"> Affine
        </label>
        </div>
        <div>
        <label>
          <input type="radio" name="curvature" value="log-log convex"> Log-Log Convex
        </label>
        <label>
          <input type="radio" name="curvature" value="log-log concave"> Log-Log Concave
        </label>
        <label>
          <input type="radio" name="curvature" value="log-log affine"> Log-Log Affine
        </label>
        </div>
      </form>
      <h5>Operation Type</h5>
      <form id="operationTypeFilter">
        <div></div>
        <label>
          <input type="radio" name="operationType" value="all" checked> All
        </label>
        <label>
          <input type="radio" name="operationType" value="scalar"> Scalar
        </label>
        <label>
          <input type="radio" name="operationType" value="elementwise"> Elementwise
        </label>
        <label>
          <input type="radio" name="operationType" value="matrix"> Matrix/Vector
        </label>
        </div>
      </form>
      <button id="applyFilters">Submit</button>
      <button id="resetFilters">Reset Filters</button>
    `;
    var dropdown = document.querySelector('.functions-filter .sd-card-text');
    if (dropdown) {
      dropdown.innerHTML = radioButtons;
    }
  }

  function initializeMainTable() {
    var table = $('table.atomic-functions').DataTable();
    $('#applyFilters').click(function() {
      var curvatureValue = $('input[name="curvature"]:checked').val();
      var operationTypeValue = $('input[name="operationType"]:checked').val();

      // Apply filters based on selected values
      if (curvatureValue !== "all" && operationTypeValue !== "all") {
        table.column(4).search('^' + curvatureValue + '$', true, false)
             .column(6).search('^' + operationTypeValue + '$', true, false).draw();
      } else if (curvatureValue !== "all") {
        table.column(4).search('^' + curvatureValue + '$', true, false).draw();
      } else if (operationTypeValue !== "all") {
        table.column(6).search('^' + operationTypeValue + '$', true, false).draw();
      }
    });

     $('#resetFilters').click(function() {
     $('input[name="curvature"][value="all"]').prop('checked', true);
     $('input[name="operationType"][value="all"]').prop('checked', true);
      table.search('').columns().search('').draw();
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

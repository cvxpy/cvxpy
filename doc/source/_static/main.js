$(document).ready(function() {
  // Function to create and insert radio buttons
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

  // Function to initialize DataTable for the main table
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

  // Call the functions to execute the code
  createRadioButtons();
  initializeMainTable();
  createDCPTable("scalar-dcp", "scalar");
  createDCPTable("element-dcp", "elementwise");

});

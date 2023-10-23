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

  // Function to create and initialize DataTable for the convex view
  function create_Scalar_DCP_Table() {
    var newTable = $('table.scalar-dcp');
    console.log(newTable)
    const scalarDCP_data = $('table.atomic-functions').DataTable().data().toArray().filter(row => {
      const curvatures = ['convex', 'concave', 'affine'];
      const curvatureValue = $(row[4]).text().trim();
      const typeValue = $(row[6]).text().trim();
      console.log(typeValue)
      return curvatures.includes(curvatureValue) && typeValue === "scalar";
    });
    console.log(scalarDCP_data)
    newTable.DataTable({
      data: scalarDCP_data,
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
  create_Scalar_DCP_Table();
});

.. _functions:

Functions
=========
The table below lists all the atomic functions available in CVXPY.

Filters
-------

.. raw:: html

    <div class="card">
        <div class="card-body">
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
            </form>
            <button id="applyFilters">Submit</button>
            <button id="resetFilters">Reset Filters</button>
        </div>
    </div>

Atoms table
-----------
.. include:: functions_table.rst
    

.. raw:: html

    <script>
    function initializeMainTable() {
        var table = $('table.atomic-functions').DataTable();
        var originalData = table.data().toArray();
        $('#operationTypeFilter input').change(function(e) {
            console.log(e)
        });
        $('#applyFilters').click(function() {
            var curvatureValue = $('input[name="curvature"]:checked').val();
            var operationTypeValue = $('input[name="operationType"]:checked').val();
            var filteredData = originalData;

            if (curvatureValue !== "all") {
                filteredData = filteredData.filter(row => {
                    const curv = $(row[4]).text().trim();
                    return curv === curvatureValue;
                });
            }
            if (operationTypeValue !== "all") {
                filteredData = filteredData.filter(row => {
                    const type = $(row[6]).text().trim();
                    return type === operationTypeValue;
                });
            }
            table.clear().rows.add(filteredData).draw();
        });

        $('#resetFilters').click(function() {
            $('input[name="curvature"][value="all"]').prop('checked', true);
            $('input[name="operationType"][value="all"]').prop('checked', true);
            table.clear().rows.add(originalData).draw();
        });
    }
    initializeMainTable();
    </script>
    
.. _functions:

Functions
=========
The table below lists all the atomic functions available in CVXPY.

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
            <h5>Type</h5>
            <form id="typeFilter">
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
            <h5>DCP Property</h5>
            <form id="dcpFilter">
                <div></div>
                <label>
                    <input type="radio" name="dcpProperties" value="all" checked> All
                </label>
                <label>
                    <input type="radio" name="dcpProperties" value="positive"> Positive
                </label>
                <label>
                    <input type="radio" name="dcpProperties" value="negative"> Negative
                </label>
                <label>
                    <input type="radio" name="dcpProperties" value="incr"> Increasing
                </label>
                <label>
                    <input type="radio" name="dcpProperties" value="decr"> Decreasing
                </label>
            </form>
        </div>
    </div>

Atoms table
-----------
.. include:: functions_table.rst
    
.. raw:: html

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.1.8/css/dataTables.dataTables.css">
    <script src="https://cdn.datatables.net/2.1.8/js/dataTables.js"></script>
    <script>
    $(document).ready(function() {
        function initializeMainTable() {
            var table = $('table.atomic-functions').DataTable( {
                            layout: {
                                topStart: 'search',
                                topEnd: null,
                                bottomStart: null,
                                bottomEnd: null,
                                bottom: 'paging'
                            }
                        } );
            var originalData = table.data().toArray();

            function applyFilters() {
                var curvatureValue = $('input[name="curvature"]:checked').val();
                var operationTypeValue = $('input[name="operationType"]:checked').val();
                var dcpFilterValue = $('input[name="dcpProperties"]:checked').val();
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
                if (dcpFilterValue !== "all") {
                    filteredData = filteredData.filter(row => {
                        const dcp = $(row[3]).text().trim();
                        return dcp.includes(dcpFilterValue);
                    });
                }
                table.clear().rows.add(filteredData).draw();
            }

            $('#curvatureFilter input, #typeFilter input, #dcpFilter input').change(applyFilters);
        }
        initializeMainTable();
    });
    </script>

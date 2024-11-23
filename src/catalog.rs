use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use crate::errors::DataFusionError;
use crate::utils::wait_for_future;
use datafusion::{
    arrow::pyarrow::ToPyArrow,
    catalog::{CatalogProvider, SchemaProvider},
    datasource::{TableProvider, TableType},
};

#[pyclass(name = "Catalog", module = "let", subclass)]
pub(crate) struct PyCatalog {
    catalog: Arc<dyn CatalogProvider>,
}

#[pyclass(name = "Database", module = "let", subclass)]
pub(crate) struct PyDatabase {
    database: Arc<dyn SchemaProvider>,
}

#[pyclass(name = "Table", module = "let", subclass)]
pub struct PyTable {
    table: Arc<dyn TableProvider>,
}

impl PyCatalog {
    pub fn new(catalog: Arc<dyn CatalogProvider>) -> Self {
        Self { catalog }
    }
}

impl PyDatabase {
    pub fn new(database: Arc<dyn SchemaProvider>) -> Self {
        Self { database }
    }
}

impl PyTable {
    pub fn new(table: Arc<dyn TableProvider>) -> Self {
        Self { table }
    }

    pub fn table(&self) -> Arc<dyn TableProvider> {
        self.table.clone()
    }
}

#[pymethods]
impl PyCatalog {
    fn names(&self) -> Vec<String> {
        self.catalog.schema_names()
    }

    #[pyo3(signature = (name="public"))]
    fn database(&self, name: &str) -> PyResult<PyDatabase> {
        match self.catalog.schema(name) {
            Some(database) => Ok(PyDatabase::new(database)),
            None => Err(PyKeyError::new_err(format!(
                "Database with name {name} doesn't exist."
            ))),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Catalog(schema_names=[{}])",
            self.names().join(";")
        ))
    }
}

#[pymethods]
impl PyDatabase {
    fn names(&self) -> HashSet<String> {
        self.database.table_names().into_iter().collect()
    }

    fn table(&self, name: &str, py: Python) -> PyResult<PyTable> {
        if let Ok(Some(table)) = wait_for_future(py, self.database.table(name)) {
            Ok(PyTable::new(table))
        } else {
            Err(DataFusionError::Common(format!("Table not found: {name}")).into())
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Database(table_names=[{}])",
            Vec::from_iter(self.names()).join(";")
        ))
    }
}

#[pymethods]
impl PyTable {
    /// Get a reference to the schema for this table
    #[getter]
    fn schema(&self, py: Python) -> PyResult<PyObject> {
        self.table.schema().to_pyarrow(py)
    }

    /// Get the type of this table for metadata/catalog purposes.
    #[getter]
    fn kind(&self) -> &str {
        match self.table.table_type() {
            TableType::Base => "physical",
            TableType::View => "view",
            TableType::Temporary => "temporary",
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let kind = self.kind();
        Ok(format!("Table(kind={kind})"))
    }
}

use pyo3::{pyclass, pymethods, PyResult};

#[pyclass(name = "Wildcard", module = "datafusion.expr", subclass)]
#[derive(Clone)]
pub struct PyWildcard {
    qualifier: Option<String>,
}

impl PyWildcard {
    pub fn new(qualifier: Option<String>) -> Self {
        Self { qualifier }
    }
}

#[pymethods]
impl PyWildcard {
    fn qualifier(&self) -> PyResult<Option<String>> {
        Ok(self.qualifier.clone())
    }
}

use datafusion_common::TableReference;
use pyo3::{pyclass, pymethods, PyResult};

#[pyclass(name = "Wildcard", module = "datafusion.expr", subclass)]
#[derive(Clone)]
pub struct PyWildcard {
    qualifier: Option<String>,
}

impl PyWildcard {
    pub fn new(qualifier: Option<TableReference>) -> Self {
        match qualifier {
            Some(reference) => Self {
                qualifier: Some((*reference.table()).to_string()),
            },
            _ => Self { qualifier: None },
        }
    }
}

#[pymethods]
impl PyWildcard {
    fn qualifier(&self) -> PyResult<Option<String>> {
        Ok(self.qualifier.clone())
    }
}

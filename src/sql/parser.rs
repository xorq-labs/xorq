use std::collections::HashMap;
use std::sync::Arc;

use crate::sql::logical::PyLogicalPlan;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::pyarrow::PyArrowType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{plan_err, Result};
use datafusion_expr::AggregateUDF;
use datafusion_expr::WindowUDF;
use datafusion_expr::{logical_plan::builder::LogicalTableSource, ScalarUDF, TableSource};
use datafusion_functions_aggregate::average::avg_udaf;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_sql::sqlparser::dialect::dialect_from_str;
use datafusion_sql::{
    planner::{ContextProvider, SqlToRel},
    sqlparser::parser::Parser,
    TableReference,
};

use pyo3::prelude::*;

fn create_table_source(fields: Vec<Field>) -> Arc<dyn TableSource> {
    Arc::new(LogicalTableSource::new(Arc::new(
        Schema::new_with_metadata(fields, HashMap::new()),
    )))
}

#[pyclass(name = "ContextProvider", module = "let", subclass)]
#[derive(Clone, Default)]
pub struct PyContextProvider {
    options: ConfigOptions,
    tables: HashMap<String, Arc<dyn TableSource>>,
}

#[pymethods]
impl PyContextProvider {
    #[pyo3(signature = (tables, config_options=None))]
    #[new]
    fn new(
        tables: HashMap<String, PyObject>,
        config_options: Option<HashMap<String, String>>,
        py: Python,
    ) -> Self {
        let mut sources = HashMap::new();
        for (name, schema) in tables.into_iter() {
            let pyarrow_schema = schema.extract::<PyArrowType<Schema>>(py);
            let fields = SchemaRef::from(pyarrow_schema.unwrap().0)
                .flattened_fields()
                .iter()
                .map(|f| Field::new(f.name(), f.data_type().clone(), f.is_nullable()))
                .collect();
            sources.insert(name, create_table_source(fields));
        }

        let mut config = ConfigOptions::new();
        if let Some(hash_map) = config_options {
            for (k, v) in &hash_map {
                config.set(k, v.clone().as_str()).unwrap();
            }
        }

        Self {
            options: config,
            tables: sources.clone(),
        }
    }
}

impl ContextProvider for PyContextProvider {
    fn get_table_source(&self, name: TableReference) -> Result<Arc<dyn TableSource>> {
        match self.tables.get(name.table()) {
            Some(table) => Ok(table.clone()),
            _ => plan_err!("Table not found: {}", name.table()),
        }
    }

    fn get_function_meta(&self, _name: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_aggregate_meta(&self, name: &str) -> Option<Arc<AggregateUDF>> {
        match name.to_lowercase().as_str() {
            "count" => Some(count_udaf()),
            "sum" => Some(sum_udaf()),
            "avg" => Some(avg_udaf()),
            _ => None,
        }
    }

    fn get_window_meta(&self, _name: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, _variable_names: &[String]) -> Option<DataType> {
        None
    }

    fn options(&self) -> &ConfigOptions {
        &self.options
    }

    fn udf_names(&self) -> Vec<String> {
        vec![]
    }

    fn udaf_names(&self) -> Vec<String> {
        vec![]
    }

    fn udwf_names(&self) -> Vec<String> {
        vec![]
    }
}

#[pyfunction]
#[pyo3(signature = (sql, context_provider, dialect=None))]
pub fn parse_sql(
    sql: &str,
    context_provider: PyContextProvider,
    dialect: Option<&str>,
) -> PyLogicalPlan {
    let dialect = match dialect {
        Some(value) => dialect_from_str(value).unwrap(),
        None => dialect_from_str("generic").unwrap(),
    };

    let ast = Parser::parse_sql(&*dialect, sql).unwrap();

    let statement = &ast[0];

    // create a context provider
    let sql_to_rel = SqlToRel::new(&context_provider);
    let plan = sql_to_rel.sql_statement_to_plan(statement.clone()).unwrap();

    PyLogicalPlan::from(plan)
}

pub(crate) fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(parse_sql))?;
    Ok(())
}

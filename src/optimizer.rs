use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::tree_node::Transformed;
use datafusion_common::DataFusionError;
use datafusion_expr::LogicalPlan;
use datafusion_optimizer::optimizer::Optimizer;
use datafusion_optimizer::{OptimizerConfig, OptimizerContext, OptimizerRule};
use pyo3::prelude::PyModule;
use pyo3::prelude::*;
use pyo3::{pyclass, pyfunction, pymethods, wrap_pyfunction, PyResult, Python};

use crate::sql::logical::PyLogicalPlan;

#[pyclass(name = "Optimizer", module = "let", subclass)]
#[derive(Clone, Default)]
pub struct PyOptimizer {
    pub optimizer: Arc<Optimizer>,
}

#[pymethods]
impl PyOptimizer {
    #[pyo3(signature = ())]
    #[new]
    pub fn new() -> Self {
        Self {
            optimizer: Arc::new(Optimizer::default()),
        }
    }
}

#[pyclass(name = "OptimizerRule", module = "let", subclass)]
#[derive(Debug)]
pub struct PyOptimizerRule {
    pub(crate) rule: PyObject,
}

unsafe impl Send for PyOptimizerRule {}
unsafe impl Sync for PyOptimizerRule {}

#[pymethods]
impl PyOptimizerRule {
    #[new]
    pub fn new(rule: &Bound<'_, PyAny>) -> Self {
        Self {
            rule: rule.clone().unbind(),
        }
    }
}

impl OptimizerRule for PyOptimizerRule {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion_common::Result<Option<LogicalPlan>> {
        self.rewrite(plan.clone(), _config).map(|o| Some(o.data))
    }

    fn name(&self) -> &str {
        "python rule"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion_common::Result<Transformed<LogicalPlan>, DataFusionError> {
        Python::with_gil(|py| {
            let py_plan = PyLogicalPlan::new(plan);
            let result = self.rule.bind(py).call_method1("try_optimize", (py_plan,));
            match result {
                Ok(py_plan) => Ok(Transformed::new_transformed(
                    py_plan
                        .extract::<PyLogicalPlan>()
                        .unwrap()
                        .plan
                        .as_ref()
                        .clone(),
                    true,
                )),
                Err(err) => Err(DataFusionError::Execution(format!("{err}"))),
            }
        })
    }
}

#[pyclass(name = "OptimizerContext", module = "let", subclass)]
#[derive(Clone, Default)]
pub struct PyOptimizerContext {
    pub(crate) context: Arc<OptimizerContext>,
}

#[pymethods]
impl PyOptimizerContext {
    #[pyo3(signature = ())]
    #[new]
    pub fn new() -> Self {
        Self {
            context: Arc::new(OptimizerContext::default()),
        }
    }
}

fn observe(_plan: &LogicalPlan, _rule: &dyn OptimizerRule) {}

#[pyfunction]
pub fn optimize_plan(plan: PyLogicalPlan, context_provider: PyOptimizerContext) -> PyLogicalPlan {
    let optimizer = Optimizer::new();
    let plan = plan.plan.as_ref().clone();
    let optimized_plan = optimizer
        .optimize(plan, context_provider.context.as_ref(), observe)
        .unwrap();
    PyLogicalPlan::from(optimized_plan)
}

pub(crate) fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(optimize_plan))?;
    Ok(())
}

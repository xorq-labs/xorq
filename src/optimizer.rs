use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{Expr, Filter, LogicalPlan, Projection};
use datafusion_optimizer::analyzer::AnalyzerRule;
use datafusion_optimizer::optimizer::Optimizer;
use datafusion_optimizer::{OptimizerConfig, OptimizerContext, OptimizerRule};
use parking_lot::RwLock;
use pyo3::prelude::PyModule;
use pyo3::prelude::*;
use pyo3::{pyclass, pyfunction, pymethods, wrap_pyfunction, PyResult, Python};

use crate::model::SessionModelRegistry;
use crate::sql::logical::PyLogicalPlan;

#[pyclass(name = "Optimizer", module = "datafusion", subclass)]
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

#[pyclass(name = "OptimizerRule", module = "datafusion", subclass)]
#[derive(Debug, Clone)]
pub struct PyOptimizerRule {
    pub(crate) rule: PyObject,
}

unsafe impl Send for PyOptimizerRule {}
unsafe impl Sync for PyOptimizerRule {}

#[pymethods]
impl PyOptimizerRule {
    #[new]
    fn new(rule: PyObject) -> Self {
        Self { rule }
    }
}

impl OptimizerRule for PyOptimizerRule {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion_common::Result<Option<LogicalPlan>> {
        Python::with_gil(|py| {
            let py_plan = PyLogicalPlan::new((*plan).clone());
            let result = self
                .rule
                .as_ref(py)
                .call_method1("try_optimize", (py_plan,));
            match result {
                Ok(py_plan) => Ok(Some(LogicalPlan::from(
                    py_plan.extract::<PyLogicalPlan>().unwrap(),
                ))),
                Err(err) => Err(DataFusionError::Execution(format!("{err}"))),
            }
        })
    }

    fn name(&self) -> &str {
        "python rule"
    }
}

#[pyclass(name = "OptimizerContext", module = "datafusion", subclass)]
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

pub struct PredictXGBoostAnalyzerRule {
    session_model_registry: Arc<RwLock<SessionModelRegistry>>,
}

unsafe impl Send for PredictXGBoostAnalyzerRule {}
unsafe impl Sync for PredictXGBoostAnalyzerRule {}

impl PredictXGBoostAnalyzerRule {
    pub(crate) fn new(session_model_registry: Arc<RwLock<SessionModelRegistry>>) -> Self {
        Self {
            session_model_registry,
        }
    }

    fn use_only_required_features(
        &self,
        fun: ScalarFunction,
        projection: Projection,
    ) -> Option<Expr> {
        let args = fun.args.clone();
        let scan = match (*projection.input).clone() {
            LogicalPlan::TableScan(scan) => Some(scan),
            LogicalPlan::Filter(Filter {
                predicate: _,
                input,
                ..
            }) => match (*input).clone() {
                LogicalPlan::TableScan(scan) => Some(scan),
                _ => None,
            },
            _ => None,
        };

        match &args[..] {
            [Expr::Literal(ScalarValue::Utf8(value)), Expr::Wildcard { qualifier: None }] => {
                if let Some(scan) = scan {
                    let model_registry = self.session_model_registry.read();
                    if let Some(model) = model_registry.models.get(value.clone().unwrap().as_str())
                    {
                        let mut args: Vec<Expr> =
                            vec![Expr::Literal(ScalarValue::from(value.clone().unwrap()))];
                        let fields = (*scan.projected_schema.fields()).clone();
                        let columns = fields.iter().enumerate().filter_map(|(index, field)| {
                            if model.feature_mapping.contains_key(&(index as i64)) {
                                return Some(Expr::Column(field.name().into()));
                            }
                            None
                        });
                        args.extend(columns);
                        return Some(Expr::ScalarFunction(ScalarFunction::new_udf(
                            fun.func.clone(),
                            args,
                        )));
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn optimize_predict(&self, projection: &Projection) -> Option<LogicalPlan> {
        let new_exprs = projection
            .expr
            .iter()
            .map(|e| {
                if let Expr::ScalarFunction(fun) = e {
                    return if fun.clone().func.name() == "predict_xgb" {
                        match self.use_only_required_features(fun.clone(), projection.clone()) {
                            Some(expr) => expr,
                            None => e.clone(),
                        }
                    } else {
                        e.clone()
                    };
                }
                e.clone()
            })
            .collect::<Vec<Expr>>();
        Projection::try_new(new_exprs, projection.input.clone())
            .ok()
            .map(LogicalPlan::Projection)
    }
}

impl AnalyzerRule for PredictXGBoostAnalyzerRule {
    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion_common::Result<LogicalPlan> {
        match plan {
            LogicalPlan::Projection(proj) => {
                self.optimize_predict(&proj)
                    .ok_or(DataFusionError::Execution(
                        "Cannot analyze plan".to_string(),
                    ))
            }
            _ => Ok(plan),
        }
    }

    fn name(&self) -> &str {
        "predict_wildcard_expansion"
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

pub(crate) fn init_module(m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(optimize_plan))?;
    Ok(())
}

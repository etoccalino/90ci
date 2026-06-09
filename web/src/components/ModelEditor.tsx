import type { Model, Shape, Variable } from '../model';
import { SHAPES } from '../model';
import { Sparkline } from './Sparkline';

interface Props {
  model: Model;
  onChange: (m: Model) => void;
  onRun: () => void;
  running: boolean;
}

export function ModelEditor({ model, onChange, onRun, running }: Props) {
  const patch = (p: Partial<Model>) => onChange({ ...model, ...p });
  const patchVar = (id: string, p: Partial<Variable>) =>
    patch({ variables: model.variables.map((v) => (v.id === id ? { ...v, ...p } : v)) });
  const addVar = () =>
    patch({
      variables: [
        ...model.variables,
        {
          id: crypto.randomUUID(),
          name: `VAR_${model.variables.length + 1}`,
          shape: 'uniform',
          p5: 0,
          p95: 1,
        },
      ],
    });
  const removeVar = (id: string) => patch({ variables: model.variables.filter((v) => v.id !== id) });

  return (
    <div className="col">
      <div className="icon-ttl">
        <div className="pgicon">ƒ</div>
        <input
          className="title-input"
          value={model.name}
          spellCheck={false}
          placeholder="Untitled model"
          onChange={(e) => patch({ name: e.target.value })}
        />
      </div>
      <div className="subtitle">Estimate a result under uncertainty · 90% confidence interval</div>

      <div className="sec-lbl">
        <span className="ico">∑</span> Model
      </div>
      <div className="formula">
        <div className="fx">ƒx</div>
        <input
          className="formula-input"
          value={model.equation}
          spellCheck={false}
          onChange={(e) => patch({ equation: e.target.value })}
        />
        <button className="runbtn" onClick={onRun} disabled={running}>
          <span className="tri">▶</span> {running ? 'Running…' : 'Run'}
        </button>
      </div>

      <div className="sec-lbl" style={{ marginTop: 26 }}>
        <span className="ico">▦</span> Random variables
      </div>
      <div className="dbtable">
        <div className="dbrow dbhead">
          <div className="cell">Name</div>
          <div className="cell">Distribution</div>
          <div className="cell">5th</div>
          <div className="cell">95th</div>
          <div className="cell">Shape</div>
        </div>
        {model.variables.map((v) => (
          <div className="dbrow" key={v.id}>
            <div className="cell title-cell">
              <input
                className="vname"
                value={v.name}
                spellCheck={false}
                onChange={(e) => patchVar(v.id, { name: e.target.value })}
              />
            </div>
            <div className="cell">
              <select
                className={`tag ${v.shape}`}
                value={v.shape}
                onChange={(e) => patchVar(v.id, { shape: e.target.value as Shape })}
              >
                {SHAPES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div className="cell">
              <input
                className="numcell num-input"
                type="number"
                value={v.p5}
                onChange={(e) => patchVar(v.id, { p5: Number(e.target.value) })}
              />
            </div>
            <div className="cell">
              <input
                className="numcell num-input"
                type="number"
                value={v.p95}
                onChange={(e) => patchVar(v.id, { p95: Number(e.target.value) })}
              />
            </div>
            <div className="cell spark-cell">
              <Sparkline shape={v.shape} />
              <button className="rm" title="Remove variable" onClick={() => removeVar(v.id)}>
                ×
              </button>
            </div>
          </div>
        ))}
        <button className="newrow" onClick={addVar}>
          <span className="p">+</span> New variable
        </button>
      </div>

      <div className="hint">
        <span className="hint-note">· 5th / 95th are the percentile bounds for each variable.</span>
      </div>
    </div>
  );
}

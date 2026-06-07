interface Props {
  modelName: string;
}

export function TopBar({ modelName }: Props) {
  return (
    <div className="topbar">
      <div className="crumb">
        <span className="sq">ƒ</span>
        <span>Models</span>
        <span className="sep">/</span>
        <span className="here">{modelName || 'Untitled model'}</span>
      </div>
    </div>
  );
}

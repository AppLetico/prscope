import { Navigate, Route, Routes } from "react-router-dom";
import { NewSessionPage } from "./pages/NewSession";
import { PlanningViewPage } from "./pages/PlanningView";
import { SessionListPage } from "./pages/SessionList";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<SessionListPage />} />
      <Route path="/new" element={<NewSessionPage />} />
      <Route path="/sessions/:id" element={<PlanningViewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

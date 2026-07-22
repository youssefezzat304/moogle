import ChatInterface from "../features/chat/components/ChatInterface";
import MoonCanvas from "../features/moon/components/MoonCanvas";
import SplitLayout from "../shared/components/layout/SplitLayout";
import "../styles/App.css";

function App() {
  return (
    <>
      <SplitLayout
        leftPanel={<MoonCanvas />}
        rightPanel={<ChatInterface />}
      ></SplitLayout>
    </>
  );
}

export default App;

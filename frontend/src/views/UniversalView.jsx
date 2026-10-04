// Task 1 — one denoising autoencoder trained on all corruption types.
import RestorationWorkspace from '../components/RestorationWorkspace.jsx';

export default function UniversalView(props) {
  return (
    <RestorationWorkspace
      {...props}
      kind="universal"
      task="Task 1"
      title="Universal Restoration"
      description="A single convolutional denoising autoencoder restores salt & pepper noise, Gaussian blur and rectangular occlusions without knowing which corruption is present."
    />
  );
}

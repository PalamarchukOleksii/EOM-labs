# EOM-labs

This repository contains several lab projects, each with its own Python scripts and dependencies, focusing on Evolutionary Optimization Methods (EOM).

## Getting Started

Follow the instructions below to set up the project.

### 1. Clone the Repository

```bash
git clone https://github.com/PalamarchukOleksii/EOM-labs.git
```

### 2. Navigate to the Project Directory

```bash
cd EOM-labs
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

- **Windows**:
  ```bash
  .\.venv\Scripts\activate
  ```

- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 5. Install Dependencies

Each lab has its own `requirements.txt` file. To install dependencies for a specific lab, navigate to that lab's folder and install its requirements.

**Example for `lab1`**:
```bash
cd lab1
pip install -r requirements.txt
```

*Note*: Repeat this process for each lab as needed.

### 6. Running the Scripts

Navigate to the folder of the lab you wish to run and execute the desired Python script.

**Example for `lab1`**:
```bash
python lab1.py
```

*Note*: Repeat this for other labs as needed.

### 7. Deactivate the Virtual Environment

Once you're done, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

- `lab1/`: First lab project
- `lab2/`: Second lab project
- ... (add more lab folders as they are created)

## Dependencies

- `venv` for virtual environment management
- Lab-specific dependencies (see each lab's `requirements.txt`)

## Licensing

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.

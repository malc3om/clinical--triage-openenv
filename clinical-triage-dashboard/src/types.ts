export interface VitalSigns {
    heart_rate: number;
    systolic_bp: number;
    diastolic_bp: number;
    spo2: number;
    respiratory_rate: number;
    temperature: number;
    gcs: number;
}

export interface Patient {
    patient_id: string;
    age: number;
    sex: string;
    chief_complaint: string;
    vitals: VitalSigns;
    vitals_trend?: Record<string, string>;
    available_labs: any[];
    pending_labs: any[];
}

export interface TriageState {
    task_id: string;
    step_number: number;
    max_steps: number;
    elapsed_minutes: number;
    patients: Patient[];
    reward: number;
    last_action_result?: string;
    last_action_error?: string;
    done: boolean;
}

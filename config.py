config = {
    # general
    'sample_rate': 24000,               
    'clip_seconds': 5.0,                
    'audio_dir': "/Users/tedwalsh/Desktop/emergesound.ai/experiments/D_JEPA/audio2",      
    'max_files': 50,                   

    # mert
    'mert_model': "m-a-p/MERT-v1-95M",    
    'mert_output_dir': "mert_embeddings", 

    # encodec
    'encodec_model_name': "facebook/encodec_24khz",
    'encodec_output_dir': "encodec_tokens",  
    'embed_dim': 768,                       

    'run_mert': True,
    'run_encodec': True,

    # denoiser 
    'num_tokens': 1024,      
    'd_model': 768,          
    'n_heads': 8,            
    't_diffusion': 1000,     
    'epochs': 500,             
    'learning_rate': 1e-4,   
    'batch_size': 4,   
          
   
    'seed': 42,

    # Model saving
    'model_save_dir': "/Users/tedwalsh/Desktop/emergesound.ai/experiments/D_JEPA/codebase_v1/models",
    'save_every': 1000, 

    # inference 

    'num_codebooks': 8,           
    'encodec_hop_length': 320,         
    'mert_frame_rate': 50,              
    'diffusion_steps': 1000,            
    'generation_dir': "/Users/tedwalsh/Desktop/emergesound.ai/experiments/D_JEPA/codebase_v1/generatedaudio", 
}

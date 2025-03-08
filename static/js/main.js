// PlateMate Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // File input preview for barcode scanner
    const fileInput = document.getElementById('image_file');
    const previewContainer = document.getElementById('image_preview');
    
    if (fileInput && previewContainer) {
        fileInput.addEventListener('change', function() {
            previewContainer.innerHTML = '';
            
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.classList.add('img-fluid', 'mt-3', 'mb-3');
                    img.style.maxHeight = '300px';
                    previewContainer.appendChild(img);
                    
                    // Show submit button after image is selected
                    const submitBtn = document.getElementById('scan_submit');
                    if (submitBtn) {
                        submitBtn.classList.remove('d-none');
                    }
                }
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }

    // Toggle password visibility
    const togglePassword = document.getElementById('togglePassword');
    const passwordField = document.getElementById('password');
    
    if (togglePassword && passwordField) {
        togglePassword.addEventListener('click', function() {
            const type = passwordField.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordField.setAttribute('type', type);
            this.classList.toggle('fa-eye');
            this.classList.toggle('fa-eye-slash');
        });
    }

    // Multi-select enhancement for allergies and health conditions
    const enhanceMultiSelect = function(selectElement, customInput) {
        if (!selectElement) return;
        
        // Add selected class to pre-selected options
        const selectedOptions = Array.from(selectElement.selectedOptions);
        selectedOptions.forEach(option => {
            option.classList.add('selected');
        });
        
        // Handle click events on options
        selectElement.addEventListener('click', function(e) {
            if (e.target.tagName === 'OPTION') {
                e.target.classList.toggle('selected');
            }
        });
        
        // Handle custom input if provided
        if (customInput) {
            const addCustomBtn = document.getElementById('add_custom_' + customInput.id);
            if (addCustomBtn) {
                addCustomBtn.addEventListener('click', function() {
                    const customValue = customInput.value.trim();
                    if (customValue) {
                        // Check if option already exists
                        let exists = false;
                        for (let i = 0; i < selectElement.options.length; i++) {
                            if (selectElement.options[i].value.toLowerCase() === customValue.toLowerCase()) {
                                exists = true;
                                selectElement.options[i].selected = true;
                                selectElement.options[i].classList.add('selected');
                                break;
                            }
                        }
                        
                        // Add new option if it doesn't exist
                        if (!exists) {
                            const newOption = document.createElement('option');
                            newOption.value = customValue;
                            newOption.text = customValue;
                            newOption.selected = true;
                            newOption.classList.add('selected');
                            selectElement.appendChild(newOption);
                        }
                        
                        // Clear the custom input
                        customInput.value = '';
                    }
                });
            }
        }
    };
    
    // Enhance allergy and health condition selects
    enhanceMultiSelect(
        document.getElementById('allergies'),
        document.getElementById('custom_allergy')
    );
    
    enhanceMultiSelect(
        document.getElementById('health_conditions'),
        document.getElementById('custom_health')
    );
}); 